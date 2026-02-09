"""
Input controller module for sending inputs to the Citra emulator.
Supports both PyAutoGUI (simpler) and Win32 API (more reliable) methods.
"""

import time
from typing import Optional, Dict, Literal
from enum import Enum

import win32gui
import win32con
import win32api
import pyautogui


class Button(Enum):
    """3DS button mappings."""
    A = "a"
    B = "b"
    X = "x"
    Y = "y"
    L = "l"
    R = "r"
    START = "start"
    SELECT = "select"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    NOOP = "noop"


class InputController:
    """
    Controls input to the Citra emulator.
    
    Supports two modes:
    - 'pyautogui': Uses PyAutoGUI for keyboard simulation (simpler but requires focus)
    - 'win32': Uses Win32 API to send messages directly to window (more reliable)
    """

    # Default Citra keyboard mappings (can be customized)
    DEFAULT_KEY_MAP = {
        Button.A: 'z',
        Button.B: 'x',
        Button.X: 'a',
        Button.Y: 's',
        Button.L: 'q',
        Button.R: 'w',
        Button.START: 'enter',
        Button.SELECT: 'backspace',
        Button.UP: 'up',
        Button.DOWN: 'down',
        Button.LEFT: 'left',
        Button.RIGHT: 'right',
        Button.NOOP: None,
    }

    # Virtual key codes for Win32 API
    VK_CODES = {
        'a': 0x41, 'b': 0x42, 'c': 0x43, 'd': 0x44, 'e': 0x45,
        'f': 0x46, 'g': 0x47, 'h': 0x48, 'i': 0x49, 'j': 0x4A,
        'k': 0x4B, 'l': 0x4C, 'm': 0x4D, 'n': 0x4E, 'o': 0x4F,
        'p': 0x50, 'q': 0x51, 'r': 0x52, 's': 0x53, 't': 0x54,
        'u': 0x55, 'v': 0x56, 'w': 0x57, 'x': 0x58, 'y': 0x59,
        'z': 0x5A,
        'enter': 0x0D, 'backspace': 0x08,
        'up': 0x26, 'down': 0x28, 'left': 0x25, 'right': 0x27,
    }

    def __init__(
        self, 
        window_title: str = "Citra",
        mode: Literal["pyautogui", "win32"] = "win32",
        key_map: Optional[Dict[Button, str]] = None
    ):
        """
        Initialize the input controller.
        
        Args:
            window_title: Partial title of the Citra window.
            mode: Input method - 'pyautogui' or 'win32'.
            key_map: Custom button-to-key mappings.
        """
        self.window_title = window_title
        self.mode = mode
        self.hwnd: Optional[int] = None
        self.key_map = key_map or self.DEFAULT_KEY_MAP.copy()
        
        # Disable PyAutoGUI failsafe for automated training
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.0
        
        # Default press duration
        self.default_duration = 0.1
        
    def find_window(self) -> bool:
        """
        Find the Citra emulator window.
        
        Returns:
            True if window found, False otherwise.
        """
        def callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if self.window_title.lower() in title.lower():
                    windows.append(hwnd)
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        
        if windows:
            self.hwnd = windows[0]
            return True
        return False
    
    def focus_window(self) -> bool:
        """
        Bring the Citra window to the foreground.
        
        Returns:
            True if successful, False otherwise.
        """
        if self.hwnd is None:
            if not self.find_window():
                return False
        
        try:
            win32gui.SetForegroundWindow(self.hwnd)
            return True
        except Exception as e:
            print(f"Error focusing window: {e}")
            return False
    
    def press_button(self, button: Button, duration: Optional[float] = None) -> bool:
        """
        Press a button for the specified duration.
        
        Args:
            button: The button to press.
            duration: How long to hold the button (seconds).
        
        Returns:
            True if successful, False otherwise.
        """
        if button == Button.NOOP:
            # No operation - just wait
            time.sleep(duration or self.default_duration)
            return True
        
        key = self.key_map.get(button)
        if key is None:
            return False
        
        duration = duration or self.default_duration
        
        if self.mode == "pyautogui":
            return self._press_pyautogui(key, duration)
        else:
            return self._press_win32(key, duration)
    
    def _press_pyautogui(self, key: str, duration: float) -> bool:
        """Press key using PyAutoGUI."""
        try:
            pyautogui.keyDown(key)
            time.sleep(duration)
            pyautogui.keyUp(key)
            return True
        except Exception as e:
            print(f"PyAutoGUI error: {e}")
            return False
    
    def _press_win32(self, key: str, duration: float) -> bool:
        """Press key using Win32 API messages."""
        if self.hwnd is None:
            if not self.find_window():
                return False
        
        vk_code = self.VK_CODES.get(key.lower())
        if vk_code is None:
            print(f"Unknown key: {key}")
            return False
        
        try:
            # Send key down
            win32api.PostMessage(self.hwnd, win32con.WM_KEYDOWN, vk_code, 0)
            time.sleep(duration)
            # Send key up
            win32api.PostMessage(self.hwnd, win32con.WM_KEYUP, vk_code, 0)
            return True
        except Exception as e:
            print(f"Win32 error: {e}")
            return False
    
    def press_action(self, action_id: int, duration: Optional[float] = None) -> bool:
        """
        Press a button by action ID (for RL action space).
        
        Action IDs:
            0: NOOP
            1: A
            2: B
            3: UP
            4: DOWN
            5: LEFT
            6: RIGHT
            7: START
            8: SELECT
            9: L
            10: R
        
        Args:
            action_id: Integer action ID.
            duration: How long to hold the button.
        
        Returns:
            True if successful, False otherwise.
        """
        action_map = {
            0: Button.NOOP,
            1: Button.A,
            2: Button.B,
            3: Button.UP,
            4: Button.DOWN,
            5: Button.LEFT,
            6: Button.RIGHT,
            7: Button.START,
            8: Button.SELECT,
            9: Button.L,
            10: Button.R,
        }
        
        button = action_map.get(action_id)
        if button is None:
            print(f"Unknown action ID: {action_id}")
            return False
        
        return self.press_button(button, duration)
    
    def set_key_mapping(self, button: Button, key: str) -> None:
        """Update the key mapping for a button."""
        self.key_map[button] = key
    
    def get_action_space_size(self) -> int:
        """Return the number of available actions."""
        return 11


if __name__ == "__main__":
    # Quick test
    print("Testing InputController module...")
    controller = InputController(mode="win32")
    
    if controller.find_window():
        print(f"Found Citra window (hwnd: {controller.hwnd})")
        
        print("Press Enter to test button presses (make sure Citra is focused)...")
        input()
        
        # Test a few button presses
        print("Testing button presses...")
        
        for button in [Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT, Button.A]:
            print(f"  Pressing {button.name}...")
            controller.press_button(button, 0.2)
            time.sleep(0.3)
        
        print("Test complete!")
    else:
        print("Could not find Citra window. Make sure Citra is running.")
