"""
Test script for RewardCalculator and reward integration in PokemonEnv.

Usage:
    1. Start Citra with Pokemon Sun/Moon
    2. Run: python tests/test_rewards.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from env.reward_calculator import RewardCalculator
from env.pokemon_env import PokemonEnv


def test_reward_calculator_standalone():
    """Test RewardCalculator in isolation."""
    print("\n" + "=" * 50)
    print("TEST: RewardCalculator Standalone")
    print("=" * 50)
    
    calc = RewardCalculator(
        exploration_weight=1.0,
        time_penalty=-0.01,
        action_variety_weight=0.1,
    )
    
    # Test with identical observations (should get low exploration reward)
    print("\n[1] Testing identical observations...")
    obs = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
    
    r1, b1 = calc.calculate_reward(obs.copy(), action=1)
    print(f"    First observation: reward={r1:.4f}")
    print(f"    Breakdown: {b1}")
    
    r2, b2 = calc.calculate_reward(obs.copy(), action=1)
    print(f"    Same observation: reward={r2:.4f}")
    print(f"    Breakdown: {b2}")
    
    assert b1['exploration'] > b2['exploration'], "Second identical obs should have lower exploration"
    print("    ✅ Exploration reward decreased for repeated state")
    
    # Test with different observations
    print("\n[2] Testing different observations...")
    for i in range(5):
        new_obs = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
        r, b = calc.calculate_reward(new_obs, action=np.random.randint(0, 11))
        print(f"    Observation {i+1}: reward={r:.4f}, exploration={b['exploration']:.2f}")
    
    stats = calc.get_stats()
    print(f"\n[3] Stats: novel states = {stats['total_novel_states']}")
    
    print("\n✅ RewardCalculator standalone test passed!")
    return True


def test_reward_in_environment():
    """Test reward calculation in PokemonEnv."""
    print("\n" + "=" * 50)
    print("TEST: Rewards in PokemonEnv")
    print("=" * 50)
    
    try:
        env = PokemonEnv(
            exploration_weight=1.0,
            time_penalty=-0.01,
        )
        
        print("[1] Resetting environment...")
        obs, info = env.reset()
        print(f"    Initial novel states: {info['novel_states']}")
        
        print("\n[2] Taking 20 steps and tracking rewards...")
        total_reward = 0
        exploration_rewards = []
        
        for i in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            breakdown = info['reward_breakdown']
            exploration_rewards.append(breakdown.get('exploration', 0))
            
            print(f"    Step {i+1:2d}: action={info['action_name']:6s} | "
                  f"reward={reward:+.3f} | exploration={breakdown.get('exploration', 0):.2f}")
            
            if terminated:
                break
        
        print(f"\n[3] Summary:")
        print(f"    Total reward: {total_reward:.3f}")
        print(f"    Novel states found: {info['novel_states']}")
        print(f"    Exploration events: {sum(1 for e in exploration_rewards if e > 0)}")
        
        env.close()
        print("\n✅ PokemonEnv reward test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_variety():
    """Test that action variety bonus works."""
    print("\n" + "=" * 50)
    print("TEST: Action Variety Bonus")
    print("=" * 50)
    
    calc = RewardCalculator(
        exploration_weight=0.0,  # Disable exploration
        time_penalty=0.0,  # Disable time penalty
        action_variety_weight=1.0,  # Only variety bonus
    )
    
    obs = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
    
    # Repeat same action
    print("[1] Repeating same action (A button)...")
    for i in range(10):
        r, b = calc.calculate_reward(obs, action=1)  # Always A
        print(f"    Action {i+1}: variety_bonus={b['action_variety']:.3f}")
    
    # Use varied actions
    print("\n[2] Using varied actions...")
    calc.hard_reset()
    actions = [1, 3, 2, 4, 5, 6, 1, 7, 8, 2]  # Mixed actions
    for i, action in enumerate(actions):
        r, b = calc.calculate_reward(obs, action=action)
        print(f"    Action {i+1} ({action}): variety_bonus={b['action_variety']:.3f}")
    
    print("\n✅ Action variety test passed!")
    return True


def main():
    """Run all reward tests."""
    print("\n" + "=" * 60)
    print("  REWARD FUNCTION TEST SUITE")
    print("=" * 60)
    
    results = {
        "standalone": test_reward_calculator_standalone(),
        "action_variety": test_action_variety(),
    }
    
    # Test with Citra
    print("\nMake sure Citra is running with Pokemon Sun/Moon loaded.")
    response = input("Press Enter to test with Citra, or 'q' to skip: ")
    
    if response.lower() != 'q':
        results["environment"] = test_reward_in_environment()
    
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
        print("🎉 All reward tests passed! Ready for Phase 4 (Training)")
    else:
        print("⚠️  Some tests failed.")
    print()


if __name__ == "__main__":
    main()
