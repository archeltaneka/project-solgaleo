import time

import cv2
import numpy as np

from src.citra.citra_environment import CitraPokemonEnv


if __name__ == "__main__":
    print("=== Citra Pokemon Environment Test ===\n")
    
    # Initialize environment
    try:
        env = CitraPokemonEnv(window_name="Citra")
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. Citra is running")
        print("2. Pokemon Sun/Moon is loaded")
        print("3. The window title contains 'Citra'")
        exit(1)
    
    print("\n=== Testing Screen Capture ===")
    screen = env.capture_screen()
    print(f"Captured screen shape: {screen.shape}")
    print(f"Screen dtype: {screen.dtype}")
    
    # Display captured screen
    cv2.imshow("Citra Screen Capture", screen)
    print("\nPress any key in the image window to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n=== Testing Button Inputs ===")
    print("The character will move in a square pattern.")
    print("Make sure Citra is in focus and you're in-game (not in menu).")
    input("Press Enter to start the movement test...")
    
    # Test movement in a square
    print("Moving UP...")
    for _ in range(3):
        env.press_button('UP', duration=0.1)
        time.sleep(0.2)
    
    print("Moving RIGHT...")
    for _ in range(3):
        env.press_button('RIGHT', duration=0.1)
        time.sleep(0.2)
    
    print("Moving DOWN...")
    for _ in range(3):
        env.press_button('DOWN', duration=0.1)
        time.sleep(0.2)
    
    print("Moving LEFT...")
    for _ in range(3):
        env.press_button('LEFT', duration=0.1)
        time.sleep(0.2)
    
    print("\n=== Testing step() Function ===")
    print("Taking random steps...")
    
    for i in range(10):
        # Random action
        action = np.random.randint(0, len(env.action_space))
        action_name = env.action_space[action]
        
        print(f"Step {i+1}: Action = {action_name}")
        observation = env.step(action)
        
        # Display
        env.render(observation)
        time.sleep(0.3)
    
    print("\n=== Test Complete ===")
    print("\nEnvironment is ready for RL training!")
    print("Action space size:", len(env.action_space))
    print("Actions:", env.action_space)
    
    env.close()
