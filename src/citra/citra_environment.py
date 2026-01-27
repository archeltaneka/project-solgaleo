"""
Citra Pokemon Sun/Moon Environment Setup
This script provides basic screen capture and input control for the Citra emulator.
"""

import numpy as np
import cv2
import time
import win32gui
import win32ui
import win32con
import ctypes
from PIL import Image
from pynput.keyboard import Key, Controller


class CitraPokemonEnv:
    """
    Environment wrapper for Pokemon Sun/Moon running in Citra emulator.
    Handles screen capture and button inputs.
    """
    
    def __init__(self, window_name="Citra"):
        """
        Initialize the environment.
        
        Args:
            window_name: The title of the Citra window (default: "Citra")
        """
        self.window_name = window_name
        self.hwnd = None
        self.keyboard = Controller()
        
        # Button mapping: Map game buttons to keyboard keys
        # Adjust these based on your Citra controller configuration
        self.button_map = {
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
        
        # Action space for RL (indices map to buttons)
        self.action_space = ['A', 'B', 'X', 'Y', 'L', 'R', 'START', 'SELECT', 
                            'UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE']
        
        print("Initializing Citra Pokemon Environment...")
        self._find_window()
        
    def _find_window(self):
        """Find the Citra emulator window."""
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if self.window_name.lower() in title.lower():
                    windows.append((hwnd, title))
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        if not windows:
            raise Exception(f"Could not find window with name '{self.window_name}'")
        
        self.hwnd = windows[0][0]
        print(f"Found window: {windows[0][1]}")
        
    def capture_screen(self, resize=None):
        """
        Capture the Citra window screen using PrintWindow.
        This method works with hardware-accelerated content (OpenGL/Vulkan).
        
        Args:
            resize: Tuple (width, height) to resize the captured image. 
                   None keeps original size.
        
        Returns:
            numpy array: RGB image of the screen
        """
        # Get the actual window size
        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        width = right - left
        height = bottom - top

        # Create DCs
        hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()

        # Create Bitmap
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bitmap)

        # Use PrintWindow instead of BitBlt
        # The '2' flag is PW_RENDERFULLCONTENT (handles hardware acceleration better)
        result = ctypes.windll.user32.PrintWindow(self.hwnd, save_dc.GetSafeHdc(), 2)

        # Convert to numpy
        bmpinfo = bitmap.GetInfo()
        bmpstr = bitmap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8)
        img = img.reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))

        # Cleanup
        win32gui.DeleteObject(bitmap.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwnd_dc)

        # Resize if requested
        if resize:
            img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)

        return img
    
    def press_button(self, button, duration=0.1):
        """
        Press a button on the emulator.
        
        Args:
            button: Button name from button_map or action_space
            duration: How long to hold the button (seconds)
        """
        if button == 'NONE':
            time.sleep(duration)
            return
        
        if button not in self.button_map:
            raise ValueError(f"Unknown button: {button}")
        
        key = self.button_map[button]
        
        # Focus the Citra window
        win32gui.SetForegroundWindow(self.hwnd)
        time.sleep(0.01)  # Small delay to ensure focus
        
        # Press and release
        self.keyboard.press(key)
        time.sleep(duration)
        self.keyboard.release(key)
        
    def step(self, action_idx, hold_duration=0.1):
        """
        Perform one step in the environment.
        
        Args:
            action_idx: Index of action in action_space
            hold_duration: How long to hold the button
            
        Returns:
            observation: Current screen state (numpy array)
        """
        if action_idx >= len(self.action_space):
            raise ValueError(f"Invalid action index: {action_idx}")
        
        action = self.action_space[action_idx]
        self.press_button(action, duration=hold_duration)
        
        # Small delay for game to respond
        time.sleep(0.05)
        
        # Capture new state - using reasonable resolution
        # 3DS native: Top=400x240, Bottom=320x240
        # We'll capture at half resolution for balance
        observation = self.capture_screen(resize=(400, 240))
        
        return observation
    
    def reset(self):
        """
        Reset the environment (you'll need to implement save state loading).
        For now, just captures the current screen.
        """
        return self.capture_screen(resize=(400, 240))
    
    def render(self, observation):
        """Display the current observation."""
        cv2.imshow('Pokemon Sun - Agent View', observation)
        cv2.waitKey(1)
        
    def close(self):
        """Cleanup resources."""
        cv2.destroyAllWindows()
