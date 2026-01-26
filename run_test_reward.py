import numpy as np

from src.citra.citra_save_state import CitraPokemonEnvWithSaveStates
from src.model.reward import RewardWrapper


if __name__ == "__main__":
    print("=== Testing Reward System ===\n")
    
    
    # Create environment with save states
    env = CitraPokemonEnvWithSaveStates(window_name="Citra")
    
    # Check available checkpoints
    print("Available checkpoints:")
    env.list_checkpoints()
    
    checkpoint_name = input("\nEnter checkpoint to test (e.g., 'route_1_start'): ").strip()
    env.set_checkpoint(checkpoint_name)
    
    # Wrap with reward system
    reward_env = RewardWrapper(env, checkpoint_type="exploration")
    
    print("\n=== Running Test Episode ===")
    print("The agent will take 50 random actions")
    print("Watch the reward values to see how the system works\n")
    
    input("Press Enter to start...")
    
    # Test episode
    obs = reward_env.reset()
    total_reward = 0
    
    for step in range(50):
        # Random action
        action = np.random.randint(0, len(env.action_space))
        action_name = env.action_space[action]
        
        # Take step
        obs, reward, done, info = reward_env.step(action)
        total_reward += reward
        
        print(f"Step {step+1}: Action={action_name:8s} | Reward={reward:+.2f} | Total={total_reward:+.2f} | Areas={info['visited_areas']}")
        
        # Display
        reward_env.render(obs)
        
        if done:
            print(f"\nEpisode ended at step {step+1}")
            break
    
    print(f"\n=== Test Complete ===")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Areas Visited: {info['visited_areas']}")
    
    reward_env.close()