import numpy as np
import torch

class NetPnlGoalManager:
    """
    Fixed goal: reach non-negative net-PnL at the end of an episode.
    Provides:
        • distance()      – scalar distance to the goal
        • relabel_rewards – HER relabelling for a whole rollout batch
    """

    def __init__(self, device):
        self.device = device

    # -------------------------------------------------------------
    # distance to goal : d = max(0 , - net_pnl)
    # -------------------------------------------------------------
    def distance(self, net_pnl):
        if isinstance(net_pnl, torch.Tensor):
            net_pnl = net_pnl.detach().cpu().numpy()
        return np.maximum(0.0, -net_pnl)

    # -------------------------------------------------------------
    # HER relabelling (called once per collect)
    # -------------------------------------------------------------
    def relabel_rewards(self, batch):
        """
        batch keys: ['rewards', 'infos']  (shape [T, B])
        Replaces reward of every time-step with Δ(–distance).
        Keeps sign so that reaching 0 PnL gives   r = +distance_prev.
        """
        rew = batch["rewards"]                    # torch Tensor [T,B]
        infos = batch["infos"]                    # list length T, each len B
        T, B = rew.shape
        # 1. compute net-PnL trajectory per env from infos
        net_pnl = torch.zeros(T, B, device=rew.device)   # realised+unrealised-fees
        for t in range(B):
            pnl_step = torch.tensor(
                [ infos[0]["realized_pnl"][t] + infos[0]["unrealized_pnl"][t] + infos[0]["fees"][t] ],
                device=rew.device, dtype=rew.dtype
            )
            net_pnl[t] = pnl_step

        # cumulative PnL until t (so final line is net PnL of episode)
        cum_pnl = net_pnl.cumsum(0)               # [T,B]

        # 2. distance to goal at every step
        dist = torch.clamp(-cum_pnl, min=0.0)     # max(0, -pnl)

        # 3. shaped reward  r'_t = dist_{t-1} − dist_t   (positive when dist shrinks)
        shaped = torch.zeros_like(rew)
        shaped[1:] = dist[:-1] - dist[1:]
        shaped[0] = -dist[0]                      # first step

        # 4. inject into batch (appended, not replacing extrinsic reward)
        batch["rewards"] = rew + shaped
