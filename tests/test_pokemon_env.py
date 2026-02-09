"""
Test script for PokemonEnv Gymnasium environment.
Runs random actions and validates the environment.

Usage:
    1. Start Citra and load Pokemon Sun/Moon
    2. Run: python tests/test_pokemon_env.py
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from env.pokemon_env import PokemonEnv
import numpy as np


def test_environment_creation():
    """Test that the environment can be created."""
    print("\n" + "=" * 50)
    print("TEST: Environment Creation")
    print("=" * 50)
    
    try:
        env = PokemonEnv()
        print(f"✅ Environment created successfully")
        print(f"   Observation space: {env.observation_space}")
        print(f"   Action space: {env.action_space}")
        print(f"   Action meanings: {env.get_action_meanings()}")
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_reset():
    """Test environment reset."""
    print("\n" + "=" * 50)
    print("TEST: Environment Reset")
    print("=" * 50)
    
    try:
        env = PokemonEnv()
        obs, info = env.reset()
        
        print(f"✅ Reset successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation dtype: {obs.dtype}")
        print(f"   Info: {info}")
        
        # Validate observation
        assert obs.shape == (84, 84, 3), f"Unexpected shape: {obs.shape}"
        assert obs.dtype == np.uint8, f"Unexpected dtype: {obs.dtype}"
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_step():
    """Test environment step."""
    print("\n" + "=" * 50)
    print("TEST: Environment Step")
    print("=" * 50)
    
    try:
        env = PokemonEnv()
        obs, info = env.reset()
        
        # Take a single step with NOOP
        obs, reward, terminated, truncated, info = env.step(0)
        
        print(f"✅ Step successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Reward: {reward}")
        print(f"   Terminated: {terminated}")
        print(f"   Info: {info}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_random_actions(num_steps: int = 20):
    """Test environment with random actions."""
    print("\n" + "=" * 50)
    print(f"TEST: Random Actions ({num_steps} steps)")
    print("=" * 50)
    
    try:
        env = PokemonEnv(frame_skip=2, action_duration=0.08)
        obs, info = env.reset()
        
        print("Running random actions...")
        step_times = []
        
        for i in range(num_steps):
            start = time.time()
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_time = time.time() - start
            step_times.append(step_time)
            
            print(f"   Step {i+1:2d}: {info['action_name']:6s} | "
                  f"reward={reward:+.2f} | time={step_time*1000:.0f}ms")
            
            if terminated:
                print("   Episode terminated!")
                break
        
        avg_step = sum(step_times) / len(step_times)
        print(f"\n✅ Random actions test complete")
        print(f"   Average step time: {avg_step*1000:.1f}ms")
        print(f"   Estimated max FPS: {1/avg_step:.1f}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def test_gymnasium_compatibility():
    """Test that the environment is compatible with Gymnasium."""
    print("\n" + "=" * 50)
    print("TEST: Gymnasium Compatibility")
    print("=" * 50)
    
    try:
        from gymnasium.utils.env_checker import check_env
        
        env = PokemonEnv()
        
        print("Running Gymnasium environment checker...")
        # This will raise an error if the environment is not compatible
        check_env(env, warn=True, skip_render_check=True)
        
        print(f"✅ Environment is Gymnasium compatible!")
        env.close()
        return True
    except Exception as e:
        print(f"⚠️  Compatibility issue: {e}")
        return False


def test_sb3_compatibility():
    """Test that the environment works with Stable-Baselines3."""
    print("\n" + "=" * 50)
    print("TEST: Stable-Baselines3 Compatibility")
    print("=" * 50)
    
    try:
        from stable_baselines3.common.env_checker import check_env
        
        env = PokemonEnv()
        
        print("Running SB3 environment checker...")
        check_env(env, warn=True)
        
        print(f"✅ Environment is SB3 compatible!")
        env.close()
        return True
    except Exception as e:
        print(f"⚠️  Compatibility issue: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  POKEMON ENVIRONMENT TEST SUITE")
    print("=" * 60)
    
    print("\nMake sure Citra is running with Pokemon Sun/Moon loaded.")
    response = input("Press Enter to start tests, or 'q' to quit: ")
    if response.lower() == 'q':
        return
    
    results = {
        "creation": test_environment_creation(),
        "reset": test_reset(),
        "step": test_step(),
        "random_actions": test_random_actions(15),
        "gymnasium": test_gymnasium_compatibility(),
        "sb3": test_sb3_compatibility(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! Ready for Phase 3 (Reward Function)")
    else:
        print("⚠️  Some tests failed or had warnings.")
    print()


if __name__ == "__main__":
    main()
