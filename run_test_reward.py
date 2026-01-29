import numpy as np
import time
from pynput import keyboard # You may need to pip install pynput
from pynput.keyboard import Key, Controller
from src.citra.citra_save_state import CitraPokemonEnvWithSaveStates
from src.model.reward import RewardWrapper
import cv2


# Mapping keyboard keys to your environment's action indices
# Adjust these strings/indices to match your env.action_space exactly
KEY_MAP = {
            'A': 'x',        # Confirm/Interact
            'B': 'z',        # Cancel/Back
            'X': 's',        # Menu
            'Y': 'a',        # 
            'L': 'q',        # L trigger
            'R': 'w',        # R trigger
            'START': Key.enter,
            'SELECT': Key.backspace,
            'UP': Key.up,
            'DOWN': Key.down,
            'LEFT': Key.left,
            'RIGHT': Key.right,
        }
print(KEY_MAP)

class ManualController:
    def __init__(self):
        self.last_action = 'NONE'
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            char = key.char
            if char in KEY_MAP:
                self.last_action = KEY_MAP[char]
        except AttributeError:
            pass

    def get_action(self):
        # We return the action then reset to NONE to prevent "stuck" keys
        action = self.last_action
        # self.last_action = 'NONE' 
        return action

if __name__ == "__main__":
    env = CitraPokemonEnvWithSaveStates(window_name="Citra")
    
    # Reverse lookup for action indices
    action_to_idx = {name: i for i, name in enumerate(env.action_space)}
    
    checkpoint_name = input("\nEnter checkpoint to test: ").strip()
    env.set_checkpoint(checkpoint_name)
    reward_env = RewardWrapper(env, checkpoint_type="battle")
    
    controller = ManualController()
    
    print("\n=== Manual Control Mode ===")
    print("Controls: W/A/S/D = Move | J = A | K = B | Q = None")
    print("Focus the Citra window, but keep this terminal visible!\n")
    
    obs = reward_env.reset()
    total_reward = 0
    step = 0

    try:
        while True:
            # 1. Get action from keyboard
            action_name = controller.get_action()
            action_idx = action_to_idx[action_name]
            
            # 2. Take step in environment
            # reward_env.calibrate_templates(env.base_env)
            obs, reward, done, info = reward_env.step(action_idx)
            total_reward += reward
            step += 1
            
            # 3. Print feedback
            # Using \r to keep the console clean and readable in real-time
            print(f"Step: {step:4d} | Last Key: {action_name:6s} | Reward: {reward:+.2f} | Total: {total_reward:+.2f} | Areas: {info['visited_areas']}", end='\r')
            
            # 4. Small sleep to match human reaction time and game speed
            time.sleep(0.1) 

            if done: break
            
    except KeyboardInterrupt:
        print("\n\nTest stopped by user.")

    print(f"\n=== Final Results ===")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Final Area Count: {info['visited_areas']}")
    reward_env.close()