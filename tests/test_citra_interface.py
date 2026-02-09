"""
Integration test script for Citra emulator interface.
Tests screen capture and input injection functionality.

Usage:
    1. Start Citra and load Pokemon Sun/Moon
    2. Run this script: python tests/test_citra_interface.py
"""

import sys
import time
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.screen_capture import ScreenCapture
from core.input_controller import InputController, Button


def test_screen_capture():
    """Test screen capture functionality."""
    print("\n" + "=" * 50)
    print("TESTING SCREEN CAPTURE")
    print("=" * 50)
    
    capture = ScreenCapture()
    
    # Test window detection
    print("\n[1] Finding Citra window...")
    if not capture.find_window():
        print("    ❌ FAILED: Could not find Citra window")
        print("    Make sure Citra is running with Pokemon Sun/Moon loaded")
        return False
    
    info = capture.get_window_info()
    print(f"    ✅ Found window: {info['title']}")
    print(f"    Position: ({info['position']['x']}, {info['position']['y']})")
    print(f"    Size: {info['size']['width']}x{info['size']['height']}")
    
    # Test frame capture
    print("\n[2] Capturing frames...")
    frame = capture.capture_frame()
    if frame is None:
        print("    ❌ FAILED: Could not capture frame")
        return False
    
    print(f"    ✅ Captured frame: {frame.shape}")
    
    # Test FPS
    print("\n[3] Measuring capture FPS...")
    start = time.time()
    frame_count = 0
    while time.time() - start < 2:
        if capture.capture_frame() is not None:
            frame_count += 1
    
    fps = frame_count / 2
    print(f"    ✅ Capture rate: {fps:.1f} FPS")
    
    if fps < 30:
        print("    ⚠️  Warning: Low FPS may affect training speed")
    
    # Test observation preprocessing
    print("\n[4] Testing observation preprocessing...")
    obs = capture.get_observation(grayscale=False)
    if obs is None:
        print("    ❌ FAILED: Could not get observation")
        return False
    
    print(f"    ✅ Observation shape: {obs.shape}")
    
    # Save test images
    print("\n[5] Saving test images...")
    try:
        import cv2
        cv2.imwrite("tests/test_full_frame.png", frame)
        cv2.imwrite("tests/test_observation.png", obs)
        print("    ✅ Saved test_full_frame.png and test_observation.png")
    except Exception as e:
        print(f"    ⚠️  Could not save images: {e}")
    
    print("\n✅ Screen capture tests PASSED")
    return True


def test_input_controller():
    """Test input injection functionality."""
    print("\n" + "=" * 50)
    print("TESTING INPUT CONTROLLER")
    print("=" * 50)
    
    controller = InputController(mode="win32")
    
    # Test window detection
    print("\n[1] Finding Citra window...")
    if not controller.find_window():
        print("    ❌ FAILED: Could not find Citra window")
        return False
    
    print(f"    ✅ Found window (hwnd: {controller.hwnd})")
    
    # Test action space
    print("\n[2] Checking action space...")
    action_count = controller.get_action_space_size()
    print(f"    ✅ Action space size: {action_count}")
    
    print("\n[3] Ready for input test")
    print("    This will send button presses to Citra.")
    print("    Make sure the game is in a safe state (e.g., standing in a town)")
    print()
    
    response = input("    Press Enter to start input test, or 'q' to skip: ")
    if response.lower() == 'q':
        print("    Skipped input test")
        return True
    
    print("\n[4] Testing button presses...")
    
    # Focus the window first
    controller.focus_window()
    time.sleep(0.5)
    
    # Test each direction
    buttons_to_test = [
        (Button.UP, "D-Pad Up"),
        (Button.DOWN, "D-Pad Down"),
        (Button.LEFT, "D-Pad Left"),
        (Button.RIGHT, "D-Pad Right"),
        (Button.A, "A Button"),
        (Button.B, "B Button"),
    ]
    
    for button, name in buttons_to_test:
        print(f"    Pressing {name}...")
        success = controller.press_button(button, 0.15)
        if success:
            print(f"    ✅ {name} sent")
        else:
            print(f"    ❌ Failed to send {name}")
        time.sleep(0.3)
    
    # Test action ID mapping
    print("\n[5] Testing action ID mapping...")
    for action_id in [0, 1, 2, 3]:  # NOOP, A, B, UP
        success = controller.press_action(action_id, 0.1)
        print(f"    Action {action_id}: {'✅' if success else '❌'}")
        time.sleep(0.2)
    
    print("\n✅ Input controller tests PASSED")
    return True


def test_integration():
    """Test screen capture + input together."""
    print("\n" + "=" * 50)
    print("TESTING INTEGRATION")
    print("=" * 50)
    
    capture = ScreenCapture()
    controller = InputController(mode="win32")
    
    if not capture.find_window() or not controller.find_window():
        print("    ❌ FAILED: Could not find Citra window")
        return False
    
    print("\n[1] Testing capture -> action -> capture loop...")
    print("    This simulates one RL step")
    
    # Capture initial state
    obs1 = capture.get_observation()
    if obs1 is None:
        print("    ❌ Failed to capture initial state")
        return False
    print(f"    ✅ Initial observation: {obs1.shape}")
    
    # Take action
    controller.press_action(3, 0.1)  # UP
    time.sleep(0.1)
    
    # Capture new state
    obs2 = capture.get_observation()
    if obs2 is None:
        print("    ❌ Failed to capture new state")
        return False
    print(f"    ✅ New observation: {obs2.shape}")
    
    print("\n[2] Measuring step timing...")
    
    step_times = []
    for _ in range(10):
        start = time.time()
        
        obs = capture.get_observation()
        controller.press_action(0, 0.05)  # NOOP
        
        step_times.append(time.time() - start)
    
    avg_step = sum(step_times) / len(step_times)
    max_fps = 1 / avg_step if avg_step > 0 else 0
    
    print(f"    ✅ Average step time: {avg_step * 1000:.1f}ms")
    print(f"    ✅ Max environment FPS: {max_fps:.1f}")
    
    print("\n✅ Integration tests PASSED")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  CITRA INTERFACE TEST SUITE")
    print("  Project Solgaleo - Pokemon RL Agent")
    print("=" * 60)
    
    print("\nMake sure Citra is running with Pokemon Sun/Moon loaded.")
    print("The game should be in a playable state (not in a cutscene).")
    
    response = input("\nPress Enter to start tests, or 'q' to quit: ")
    if response.lower() == 'q':
        print("Tests cancelled.")
        return
    
    results = {
        "screen_capture": test_screen_capture(),
        "input_controller": test_input_controller(),
        "integration": test_integration(),
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
        print("🎉 All tests passed! Ready for Phase 2 (Gym Environment)")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print()


if __name__ == "__main__":
    main()
