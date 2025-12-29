from dataclasses import dataclass

@dataclass
class HierarchicalConfig:
    num_envs: int = 8
    num_threads: int = 8
    n_steps: int = 3600*2   # 2 hours 
    max_episode_steps: int = 3600*2-1 # 2 hours - 1 step

    inventory_update_freq: int = 1  # Every step (target is smoothed by 120-step EMA in strategy)

    # PPO hyperparameters
    learning_rate: float = 1e-3
    inv_learning_rate: float = 1e-3  # Reduced from 4e-5 to stabilize (large gradients causing instability)
    gamma: float = 0.5  # Lower gamma for more immediate rewards
    gae_lambda: float = 0.90  # GAE lambda for advantage estimation (less bootstrap than usual 0.95)
    gae_clip_delta: float = 2.0  # Clip TD errors in GAE computation
    normalize_gae: bool = True  # Normalize advantages after GAE computation
    clip_range: float = 0.2
    entropy_coef: float = 0.5  # Base entropy coefficient (reduced to prevent std explosion)
    entropy_coef_mm: float = 0.5  # Higher entropy for MM agent (encourages exploration, small policy loss suggests early convergence)
    entropy_coef_inv: float = 0.5  # Lower entropy for inventory agent (already learning actively)
    value_coef: float = 0.2
    max_grad_norm: float = 0.5
    value_loss_clip: float = 0.5  # Clip value loss to prevent overfitting (max MSE allowed)
    value_l2_reg: float = 1e-4  # L2 regularization for value function (prevents overfitting)
    update_epochs: int = 4
    minibatch_size: int = 256

    # Trading parameters
    base_spread_bps: float = 1
    min_size_pct: float = 1
    max_size_pct: float = 25.0
    balance: float = 20000.0
    target_range: float = 1.0  # Inventory agent outputs actions in [-target_range, +target_range], used directly in C++

    # Training
    total_epochs: int = 10000
    save_interval: int = 500000
    log_interval: int = 1

