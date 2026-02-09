"""
Demo script showing how to use PokemonEnv with Stable-Baselines3.
This is a minimal example that trains for just a few steps.

For actual training, see the training scripts in src/agent/
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.pokemon_env import PokemonEnv


def main():
    print("=" * 60)
    print("  POKEMON RL DEMO - SB3 Integration")
    print("=" * 60)
    
    print("\nCreating environment...")
    
    # Wrap environment for SB3
    env = DummyVecEnv([lambda: PokemonEnv(
        obs_width=84,
        obs_height=84,
        frame_skip=4,
        action_duration=0.1
    )])
    
    print("✅ Environment wrapped in DummyVecEnv")
    
    # Create PPO agent
    print("\nCreating PPO agent...")
    model = PPO(
        "CnnPolicy",  # CNN policy for image observations
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        device="cuda",  # Use GPU
        tensorboard_log="./logs/pokemon_tensorboard/"
    )
    
    print("✅ PPO agent created")
    print(f"   Policy: {model.policy.__class__.__name__}")
    print(f"   Device: {model.device}")
    
    # Demo: just take a few steps
    print("\nDemo: Taking 10 steps with untrained agent...")
    obs = env.reset()
    
    for i in range(10):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, info = env.step(action)
        print(f"   Step {i+1}: action={action[0]}, reward={reward[0]:.2f}")
    
    print("\n✅ Demo complete!")
    print("\nTo train for real, you would call:")
    print("   model.learn(total_timesteps=100000)")
    print("\nBut first, implement the reward function in Phase 3!")
    
    env.close()


if __name__ == "__main__":
    input("Press Enter to start (make sure Citra is running)...")
    main()
