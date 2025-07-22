#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bayesian inference script that mirrors the hyper-parameters and environment
settings used in *tianshou_ppo.py*.
"""

from pathlib import Path
import numpy as np
import torch
import litepool

from vec_normalizer import VecNormalize
from recurrent_actor_critic import RecurrentActorCritic
from recurrent_ppo_policy import RecurrentPPOPolicy
from metric_logger import MetricLogger

# --------------------------------------------------------------------------- #
# Globals                                                                     #
# --------------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_of_envs = 1
max_steps = 3600 * 400          # identical horizon to training (400 hours)

# --------------------------------------------------------------------------- #
# Safe VecNormalize                                                           #
# --------------------------------------------------------------------------- #
class SafeVecNormalize(VecNormalize):
    """Same as VecNormalize but avoids in-place operations during reset."""
    def reset(self, env_id=None):
        if env_id is None:
            obs, info = self.env.reset()
            self.returns = torch.zeros_like(self.returns)
        else:
            env_id = np.atleast_1d(env_id)
            obs, info = self.env.reset(env_id)
            self.returns[env_id] = 0.0

        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        obs = self.normalize_obs(obs)
        return obs.cpu().numpy(), info


# --------------------------------------------------------------------------- #
# Utility functions                                                           #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_bayesian_action(model, obs_t, hidden_state, n_samples: int = 10):
    """
    Monte-Carlo sample `n_samples` actions, return mean & std of the samples.
    Works with any forward signature as long as:
        • outs[0] = action distribution
        • outs[-1] = next hidden state
    """
    acts = []
    new_hidden = hidden_state
    for _ in range(n_samples):
        outs = model(obs_t, hidden_state)
        dist, new_hidden = outs[0], outs[-1]
        acts.append(torch.tanh(dist.rsample()))
    acts = torch.stack(acts)           # [n, batch, act_dim]
    return acts.mean(0), acts.std(0), new_hidden


def extract_pnl(info):
    """Safely pull scalar PnL, fees, leverage from info dict."""
    if isinstance(info, list):
        info = info[0]

    def _get(k):
        v = info.get(k, 0.0)
        return float(v[0]) if isinstance(v, (list, np.ndarray)) else float(v)

    return {k: _get(k) for k in
            ("realized_pnl", "unrealized_pnl", "fees", "leverage")}


# --------------------------------------------------------------------------- #
# Load env & model                                                            #
# --------------------------------------------------------------------------- #
def load_model_and_env():
    env = litepool.make(
        "RlTrader-v0",
        env_type="gymnasium",
        num_envs=num_of_envs,
        batch_size=num_of_envs,
        num_threads=1,
        is_prod=True,
        is_inverse_instr=True,
        api_key="Wd82oXzXWzJRf4ZHpsijXK", # PROD
        api_secret="cxakp_SE3VYcB2u336cdcfN5hiBU", #PROD
        #api_key="pd4khWVrKGgRL8HHgbDs1X", # UAT 
        #api_secret="cxaks_gCz5k6k9ucFUB4dQASiKEB", #UAT
        symbol="BTCUSD-PERP",
        hedge_symbol="BTC",
        tick_size=0.1,
        min_amount=0.0001,
        maker_fee=-0.00004,          # === matches training script ===
        taker_fee=0.0005,
        foldername="./test_files/",
        balance=0.05,
        start=1,
        max=max_steps,
    )
    env.spec.id = "RlTrader-v0"

    env = SafeVecNormalize(
        env,
        device=device,
        num_envs=num_of_envs,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.95,                 # === matches training script ===
    )

    # load normalisation statistics
    results_dir = Path("results")
    vec_path = results_dir / "vecnorm.pth"
    if vec_path.exists():
        env.load(vec_path)
        print(f"[VecNorm] Loaded statistics from {vec_path}")
    else:
        print(f"[VecNorm] WARNING: stats file not found → {vec_path}")

    # model
    model = RecurrentActorCritic(
        action_dim=env.action_space.shape[0],
        hidden_dim=128,
        gru_hidden_dim=128,
        num_layers=2,
    ).to(device)
    model.eval()

    model_path = results_dir / "final_model_inference.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"[Model] Loaded weights from {model_path}")

    # make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return env, model


# --------------------------------------------------------------------------- #
# Main loop                                                                   #
# --------------------------------------------------------------------------- #
def main():
    env, model = load_model_and_env()
    policy = RecurrentPPOPolicy(model=model)          # only for hidden states
    logger = MetricLogger(print_interval=512)

    obs, info = env.reset()
    print("Env reset from infer_ppo")
    hidden = policy.init_hidden_state(batch_size=num_of_envs)
    hidden = tuple(h.to(device) for h in hidden) if isinstance(hidden, tuple) else hidden.to(device)

    step, cum_reward = 0, 0.0
    ep_rewards, ep_infos, ep_len = [], [], 0

    print("\n=== Bayesian PPO Inference ===")
    print("Step |  R_t  | ΣR |  RPnL | UPnL | Fees | Lev | σ(a)")

    try:
        while step < max_steps:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mean_a, std_a, hidden = sample_bayesian_action(model, obs_t, hidden, n_samples=10)

            next_obs, reward, terminated, truncated, info = env.step(mean_a.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            r = float(reward[0] if isinstance(reward, np.ndarray) else reward)
            cum_reward += r
            ep_rewards.append(r)
            ep_infos.append(info)
            ep_len += 1
            step += 1

            pnl = extract_pnl(info)
            print(f"{step:6d} | {r:+5.3f} | {cum_reward:+6.2f} | "
                  f"{pnl['realized_pnl']:+6.4f} | {pnl['unrealized_pnl']:+6.4f} | "
                  f"{pnl['fees']:+6.4f} | {pnl['leverage']:4.2f}x | "
                  f"{std_a.mean().item():5.4f}", end="\r")

            if np.any(done):
                logger.log(step, {"infos": ep_infos, "episode_length": ep_len},
                           np.array(ep_rewards), policy)
                print(f"\n[Episode End] len={ep_len}  ΣR={sum(ep_rewards):.2f}")
                obs, info = env.reset()
                hidden = policy.init_hidden_state(batch_size=num_of_envs)
                hidden = tuple(h.to(device) for h in hidden) if isinstance(hidden, tuple) else hidden.to(device)
                ep_rewards, ep_infos, ep_len = [], [], 0
            else:
                obs = next_obs

    except KeyboardInterrupt:
        print("\nInference interrupted.")

    finally:
        if ep_rewards:
            logger.log(step, {"infos": ep_infos, "episode_length": ep_len},
                       np.array(ep_rewards), policy)
        print("\n=== Inference Complete ===")
        print(f"Total steps: {step}")
        print(f"Cumulative reward: {cum_reward:.2f}")
        env.close()


if __name__ == "__main__":
    main()
