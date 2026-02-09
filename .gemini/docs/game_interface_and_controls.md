## Game Interface & Controls

### Input Injection Methods

**Option 1: PyAutoGUI (Simpler, Less Reliable)**
```python
import pyautogui
import time

class CitraController:
    def __init__(self):
        self.button_map = {
            'A': 'z',
            'B': 'x',
            'X': 'a',
            'Y': 's',
            'L': 'q',
            'R': 'w',
            'START': 'enter',
            'SELECT': 'backspace',
            'D_UP': 'up',
            'D_DOWN': 'down',
            'D_LEFT': 'left',
            'D_RIGHT': 'right'
        }
    
    def press_button(self, button, duration=0.1):
        """Simulate button press"""
        if button in self.button_map:
            pyautogui.keyDown(self.button_map[button])
            time.sleep(duration)
            pyautogui.keyUp(self.button_map[button])
    
    def focus_window(self):
        """Ensure Citra window is focused"""
        # Use pyautogui to click on Citra window
        # Or use win32gui to bring window to front
        pass
```

**Option 2: Direct Window Message (More Reliable)**
```python
import win32gui
import win32con
import win32api

class CitraControllerAdvanced:
    def __init__(self):
        self.hwnd = None
        self.find_citra_window()
        
        # Virtual key codes for Citra's default controls
        self.vk_map = {
            'A': 0x5A,      # Z key
            'B': 0x58,      # X key
            'UP': 0x26,     # Arrow up
            'DOWN': 0x28,   # Arrow down
            'LEFT': 0x25,   # Arrow left
            'RIGHT': 0x27   # Arrow right
        }
    
    def find_citra_window(self):
        """Find Citra window handle"""
        def callback(hwnd, windows):
            if "Citra" in win32gui.GetWindowText(hwnd):
                windows.append(hwnd)
            return True
        
        windows = []
        win32gui.EnumWindows(callback, windows)
        if windows:
            self.hwnd = windows[0]
    
    def send_key(self, key, press_time=0.1):
        """Send key press to Citra window"""
        if self.hwnd and key in self.vk_map:
            # WM_KEYDOWN
            win32api.PostMessage(self.hwnd, win32con.WM_KEYDOWN, 
                                self.vk_map[key], 0)
            time.sleep(press_time)
            # WM_KEYUP
            win32api.PostMessage(self.hwnd, win32con.WM_KEYUP, 
                                self.vk_map[key], 0)
```

### Screen Capture

**Fast Screen Capture with MSS:**
```python
import mss
import numpy as np
from PIL import Image

class ScreenCapture:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = None
        self.setup_monitor_region()
    
    def setup_monitor_region(self):
        """Define capture region (Citra game window)"""
        # You'll need to detect or manually set Citra window coordinates
        self.monitor = {
            "top": 100,    # Adjust based on window position
            "left": 100,
            "width": 400,  # Native 3DS resolution
            "height": 480  # Top + bottom screens
        }
    
    def capture_frame(self):
        """Capture current frame as numpy array"""
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        return frame[:, :, :3]  # Remove alpha channel
    
    def capture_top_screen(self):
        """Capture only top screen (main gameplay)"""
        frame = self.capture_frame()
        # Top screen is 400x240, adjust if using higher resolution
        return frame[:240, :, :]
```