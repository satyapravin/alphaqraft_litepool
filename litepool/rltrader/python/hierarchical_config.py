from dataclasses import dataclass

@dataclass
class HierarchicalConfig:
    num_envs: int = 8
    num_threads: int = 8
    n_steps: int = 3600  
    max_episode_steps: int = 3600-1 

    inventory_update_freq: int = 1  # Every step (target is smoothed by 120-step EMA in strategy)

    # PPO hyperparameters
    learning_rate: float = 1e-3
    inv_learning_rate: float = 1e-3  # Same as MM agent LR for consistency
    gamma: float = 0.99  # Discount factor (high for long-term credit assignment)
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation
    gae_clip_delta: float = 2.0  # Clip TD errors in GAE computation
    normalize_gae: bool = True  # Normalize advantages after GAE computation (uses robust normalization)
    clip_range: float = 0.2  # PPO clipping range
    entropy_coef_mm: float = 0.5  # Entropy bonus for MM agent (encourages spread exploration)
    entropy_coef_inv: float = 1.0  # Higher entropy for inventory agent (encourages position exploration)
    value_coef: float = 0.2  # Value loss coefficient
    max_grad_norm: float = 10.0  # Gradient clipping norm (increased from 0.5 to prevent value loss explosion)
    value_l2_reg: float = 1e-4  # L2 regularization for value function
    update_epochs: int = 4  # PPO update epochs per rollout
    minibatch_size: int = 256  # Minibatch size for PPO updates

    # Trading parameters
    # Base spread in basis points (1 bps = 0.01% of mid price)
    # This is the MINIMUM spread - agent can only widen, not tighten below this
    # For BTC at $100k: 2 bps = $20 total spread
    base_spread_bps: float = 0.1  # Wider spread to reduce adverse selection (was 1)
    min_size_pct: float = 1
    max_size_pct: float = 25.0
    balance: float = 10000.0
    target_range: float = 1 # Inventory agent outputs actions in [-target_range, +target_range], used directly in C++

    # Training
    total_epochs: int = 10000
    save_interval: int = 100  # Save checkpoint every 100 epochs (was 500000, never triggered)
    log_interval: int = 1

