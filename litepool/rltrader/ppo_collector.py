import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
import time

# === Vectorised OT helpers ===============================================
def pairwise_gaussian_kl(mean, log_std):
    """
    KL(i||j) between batches of diagonal Gaussians.
    mean/log_std: [B, T, A]  (env, time, action_dim)
    returns      : [B, T, T]
    """
    var_i = (2 * log_std).exp()                       # [B,T,A]
    var_j = var_i.unsqueeze(2)                        # broadcast
    diff   = mean.unsqueeze(2) - mean.unsqueeze(1)    # [B,T,T,A]

    kl = (
        log_std.unsqueeze(2) - log_std.unsqueeze(1) +
        (var_i.unsqueeze(2) + diff.pow(2)) / (2 * var_j) - 0.5
    ).sum(-1)                                         # -> [B,T,T]
    return kl


def sinkhorn_log(cost, eps=0.05, n_iter=20):
    """
    Log-domain Sinkhorn – numerically stable and GPU-friendly.
    cost : [B, T, T] (non-negative)
    """
    log_K = -cost / eps                               # log kernel
    log_u = log_v = torch.zeros_like(cost[:, :, 0])   # [B,T]

    for _ in range(n_iter):
        log_u = -torch.logsumexp(log_K + log_v.unsqueeze(1), dim=-1)
        log_v = -torch.logsumexp(log_K + log_u.unsqueeze(2), dim=-2)

    # transport plan Π = diag(u) K diag(v)  in log-space
    return (log_u.unsqueeze(2) + log_v.unsqueeze(1) + log_K).exp()
# ==========================================================================

class PPOCollector:
    def __init__(self, env, policy, n_steps, gamma=0.99, gae_lambda=0.95, device="cuda", use_ot=True, ot_reg=0.03):
        self.env = env
        self.policy = policy
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.last_obs = None
        self.last_hidden_state = None
        self.use_ot = use_ot
        self.ot_reg = ot_reg  # Regularization parameter for Sinkhorn

    # ======================================================================
    # PPOCollector.collect
    # ----------------------------------------------------------------------
    def collect(self):
        """
        Roll out the current policy for `self.n_steps` time–steps on
        `self.env.num_envs` parallel environments and return a dictionary
        containing everything the optimiser needs.

        Major differences versus the original implementation:
        1.  All tensors stay on GPU – no CPU ⇆ GPU ping-pong.
        2.  No per-environment Python loop during OT/GAE computation.
        3.  Hidden states, observations and rewards are stacked directly
            on the device to minimise memory traffic.
        """
        n_envs   = self.env.num_envs
        device   = self.device

        # ------------------------------------------------------------------
        # Allocate trajectory containers (lists are faster than pre-sized
        # tensors for varying hidden state tuples)
        # ------------------------------------------------------------------
        batch_obs,   batch_actions   = [], []
        batch_logps, batch_values    = [], []
        batch_rewards, batch_dones   = [], []
        batch_infos,  batch_states   = [], []

        # ------------------------------------------------------------------
        # Prepare first observation + RNN state
        # ------------------------------------------------------------------
        if self.last_obs is None:
            obs, info          = self.env.reset()
            hidden_state       = self.policy.init_hidden_state(batch_size=n_envs)
        else:
            obs           = self.last_obs
            hidden_state   = self.last_hidden_state

        # move hidden_state to GPU
        hidden_state = self._to_device(hidden_state)

        # ------------------------------------------------------------------
        #  Roll-out loop
        # ------------------------------------------------------------------
        for _ in tqdm(range(self.n_steps)):
            obs_tensor = torch.as_tensor(obs,
                                         dtype=torch.float32,
                                         device=device)            # [B,obs]

            # --- policy forward ------------------------------------------
            action, log_prob, value, entropy, next_hidden_state = \
                self.policy.forward(obs_tensor, hidden_state)

            # --- env step (env expects numpy on host) --------------------
            next_obs, reward, done, trunc, info = \
                self.env.step(action.detach().cpu().numpy())

            # --- store trajectory ----------------------------------------
            batch_obs.append(obs_tensor)
            batch_actions.append(action.detach())
            batch_logps.append(log_prob.detach())
            batch_values.append(value.detach())
            batch_rewards.append(torch.as_tensor(reward,
                                                 dtype=torch.float32,
                                                 device=device))
            batch_dones.append(torch.as_tensor(done,
                                               dtype=torch.float32,
                                               device=device))
            batch_infos.append(info)
            batch_states.append(tuple(h.detach() for h in hidden_state))

            # --- reset envs that finished -------------------------------
            finished = np.logical_or(done, trunc)
            if finished.any():
                idx = np.where(finished)[0]
                for env_id in idx:
                    reset_obs, _ = self.env.reset(env_id)
                    next_obs[env_id] = reset_obs
                    # zero corresponding hidden units
                    if isinstance(hidden_state, tuple):
                        for h in hidden_state:
                            h[:, env_id].zero_()
                        for h in next_hidden_state:
                            h[:, env_id].zero_()
                    else:                           # single GRU tensor
                        hidden_state[:, env_id].zero_()
                        next_hidden_state[:, env_id].zero_()

            # --- advance -------------------------------------------------
            obs, hidden_state = next_obs, next_hidden_state

        # Save last obs / state for the next rollout
        self.last_obs          = torch.as_tensor(obs, dtype=torch.float32)
        self.last_hidden_state = tuple(h.detach().clone()
                                       for h in hidden_state)

        # ------------------------------------------------------------------
        # Bootstrap value for final observations
        # ------------------------------------------------------------------
        with torch.no_grad():
            final_obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
            _, _, next_value, _, _ = self.policy.forward(final_obs, hidden_state)
        next_value = next_value.detach()                                 # [B]

        # ------------------------------------------------------------------
        # Stack lists into tensors – all remain on GPU
        # ------------------------------------------------------------------
        batch_obs       = torch.stack(batch_obs,   dim=0)   # [T,B,obs]
        batch_actions   = torch.stack(batch_actions, dim=0) # [T,B,act]
        batch_logps     = torch.stack(batch_logps,   dim=0) # [T,B]
        batch_values    = torch.stack(batch_values,  dim=0) # [T,B]
        batch_rewards   = torch.stack(batch_rewards, dim=0) # [T,B]
        batch_dones     = torch.stack(batch_dones,   dim=0) # [T,B]
        # hidden-state tuple: list→tensor, keep device
        batch_states = tuple(torch.stack([s[i] for s in batch_states],
                                         dim=0)             # [T,L,B,H]
                             for i in range(len(batch_states[0])))

        # ------------------------------------------------------------------
        # Compute OT-based advantages & returns (vectorised)
        # ------------------------------------------------------------------
        advantages, returns = self._compute_advantages(
            rewards=batch_rewards,
            values=batch_values,
            dones=batch_dones,
            next_value=next_value,
            obs=batch_obs,
            states=batch_states
        )

        # Detach to avoid holding computation graph
        advantages, returns = advantages.detach(), returns.detach()

        # ------------------------------------------------------------------
        # Pack everything into a Batch-dict
        # ------------------------------------------------------------------
        batch = {
            "obs"       : batch_obs,         # [T,B,obs_dim]
            "actions"   : batch_actions,     # [T,B,act_dim]
            "log_probs" : batch_logps,       # [T,B]
            "values"    : batch_values,      # [T,B]
            "rewards"   : batch_rewards,     # [T,B]
            "dones"     : batch_dones,       # [T,B]
            "advantages": advantages,        # [T,B]
            "returns"   : returns,           # [T,B]
            "infos"     : batch_infos,       # list(len=T) of env info
            "states"    : batch_states       # tuple(seq,layer,B,H)
        }
        return batch
    # ======================================================================

    def compute_sinkhorn_plan(self, cost_matrix, reg=0.01, max_iter=50):
        """
        Compute the Sinkhorn transport plan for OT.
        Args:
            cost_matrix: [n, m] cost matrix (e.g., KL divergence between action distributions)
            reg: Entropy regularization parameter
            max_iter: Maximum number of Sinkhorn iterations (reduced for speed)
        Returns:
            transport_plan: [n, m] transport plan matrix
        """
        # Normalize cost matrix
        cost_matrix = cost_matrix / (cost_matrix.max() + 1e-8)
        
        # Initialize dual variables
        n, m = cost_matrix.shape
        u = torch.ones(n, device=self.device) / n
        v = torch.ones(m, device=self.device) / m
        
        # Precompute kernel
        K = torch.exp(-cost_matrix / reg)
        
        # Sinkhorn iterations
        for _ in range(max_iter):
            u_new = 1.0 / (n * torch.matmul(K, v))
            v = 1.0 / (m * torch.matmul(K.t(), u_new))
            u = u_new
        
        # Compute transport plan
        transport_plan = torch.diag(u) @ K @ torch.diag(v)
        return transport_plan

    # -------------------------------------------------------------------------
    def _compute_advantages(self, rewards, values, dones,
                            next_value, obs, states):
        """
        Vectorised OT-GAE; no Python loops over envs.
        All tensors are already on GPU.
        """
        B, T = rewards.shape[1], rewards.shape[0]

        # ---------- 1. conditional action distributions ----------------------
        with torch.no_grad():
            init_state = tuple(s[0] for s in states)          # [L,B,H] per GRU
            dist, _, _, _ = self.policy.forward_train(obs, init_state)
            mean     = dist.mean               # [T, B, A]
            log_std  = dist.stddev.log()       # [T, B, A]

            # move time in second dimension -> [B, T, A]
            mean     = mean.permute(1, 0, 2).contiguous()
            log_std  = log_std.permute(1, 0, 2).contiguous()

            cost   = pairwise_gaussian_kl(mean, log_std)   # [B, T, T]
            plan   = sinkhorn_log(cost, eps=self.ot_reg)   # [B, T, T]

        # ---------- 2. OT-weighted returns -----------------------------------
        gamma = self.gamma
        discounts = torch.logspace(0, T-1, steps=T, base=gamma,
                                   device=self.device).view(1, 1, T)

        # R̃_t  = Σ_{τ≥t} Π_{t,τ}  ·  (γ^{τ-t} r_τ)
        ot_rewards = torch.einsum("bij,bj->bi",
                                  plan * discounts, rewards.permute(1,0))  # [B,T]

        # boot-strap the value function
        last_values = torch.cat([values[1:], next_value.unsqueeze(0)], dim=0)

        deltas = rewards + gamma * last_values * (1 - dones) - values
        gae    = torch.zeros_like(deltas[0])

        advantages = torch.zeros_like(deltas)
        for t in reversed(range(T)):
            gae = deltas[t] + gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        advantages = advantages + ot_rewards.permute(1, 0)  # merge
        returns     = advantages + values

        # ---------- 3. normalise ---------------------------------------------
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.detach(), returns.detach()
    # -------------------------------------------------------------------------

    def _compute_gae(self, rewards, values, dones, next_value):
        """
        Original GAE computation (unchanged).
        """
        n_steps, n_envs = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(n_envs, dtype=torch.float32)

        for step in reversed(range(n_steps)):
            if step == n_steps - 1:
                next_vals = next_value
                next_non_terminal = 1.0 - dones[step]
            else:
                next_vals = values[step + 1]
                next_non_terminal = 1.0 - dones[step]

            delta = rewards[step] + self.gamma * next_vals * next_non_terminal - values[step]
            advantages[step] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + values
        return advantages, returns

    def _to_device(self, hidden_state):
        if isinstance(hidden_state, tuple):
            return tuple(h.to(self.device) for h in hidden_state)
        else:
            return hidden_state.to(self.device)
