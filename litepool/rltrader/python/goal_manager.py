import collections
from typing import Dict, List, Any
import numpy as np
import torch


class NetPnlGoalManager:
    """
    HER-style reward shaper for the fixed goal:
        Net-PnL (realised + unrealised − fees) ≥ 0   per EPISODE.

    • Works when an episode spans many rollouts (e.g. 24 h).
    • Keeps one scalar distance d_{t-1} for every parallel environment
      so the distance signal is continuous across rollout boundaries.
    """

    def __init__(self, device: torch.device):
        self.device = device
        # env_id -> last distance value  (resets to 0 at episode end)
        self.prev_dist: Dict[int, float] = collections.defaultdict(float)

    def _safe_get(self, info: dict, key: str, env_id: int, default: float = 0.0) -> float:
        """Safely extract value from info dict, handling arrays and scalars."""
        val = info.get(key)
        if val is None:
            return default
        if isinstance(val, (np.ndarray, list)):
            if env_id < len(val):
                return float(val[env_id])
            return default
        return float(val)

    # ------------------------------------------------------------------
    # Main entry point – call once immediately after `collector.collect`
    # ------------------------------------------------------------------
    def relabel_rewards(self, batch: Dict[str, Any]) -> None:
        """
        Adds shaped rewards to the existing reward tensor in-place.

        batch keys used:
            • rewards : torch.Tensor [T, B]
            • dones   : torch.Tensor [T, B] - done flags from env.step()
            • infos   : List[dict] length T, each dict has arrays indexed by env_id
        """
        rew   = batch["rewards"]          # [T, B]  torch tensor
        dones = batch["dones"]            # [T, B]  torch tensor
        infos = batch["infos"]            # Python list, len T
        T, B  = rew.shape
        shaped = torch.zeros_like(rew)

        # ------------------------------------------------------------------
        # Loop over parallel environments, compute shaped reward separately
        # ------------------------------------------------------------------
        for env_id in range(B):
            # ---- 1. extract per-step Net-PnL trajectory ------------------
            pnl_list: List[float] = []
            done_flags: List[bool] = []

            for t in range(T):
                info_t = infos[t] if t < len(infos) else {}
                if not isinstance(info_t, dict):
                    info_t = {}
                
                realized = self._safe_get(info_t, "realized_pnl", env_id, 0.0)
                unrealized = self._safe_get(info_t, "unrealized_pnl", env_id, 0.0)
                fees = self._safe_get(info_t, "fees", env_id, 0.0)
                
                net = realized + unrealized - fees
                pnl_list.append(net)
                done_flags.append(bool(dones[t, env_id]))

            pnl = torch.tensor(pnl_list, dtype=rew.dtype, device=self.device)  # [T]

            # ---- 2. distance to goal at every step ------------------------
            dist = torch.clamp(-pnl, min=0.0)                                  # d_t

            # ---- 3. delta-distance shaped rewards -------------------------
            d_prev = torch.as_tensor(self.prev_dist[env_id],
                                     dtype=rew.dtype, device=self.device)

            shaped[0, env_id]  = d_prev - dist[0]          # first step
            shaped[1:, env_id] = dist[:-1] - dist[1:]      # rest

            # ---- 4. remember last distance for next rollout ---------------
            self.prev_dist[env_id] = dist[-1].item()

            # ---- 5. reset tracker if episode ended inside this rollout ----
            if any(done_flags):
                self.prev_dist[env_id] = 0.0

        # ------------------------------------------------------------------
        # 6. inject shaped rewards (additive shaping)
        # ------------------------------------------------------------------
        batch["rewards"] = rew + shaped
