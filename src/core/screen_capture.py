"""
Screen capture module for capturing frames from Citra emulator window.
Provides fast screen capture using MSS for real-time RL training.
"""

import numpy as np
import mss
import cv2
import win32gui
from typing import Optional, Tuple, Dict


class ScreenCapture:
    """Captures game frames from the Citra emulator window."""

    def __init__(self, window_title: str = "Citra"):
        """
        Initialize the screen capture module.
        
        Args:
            window_title: Partial title of the Citra window to capture.
        """
        self.window_title = window_title
        self.sct = mss.mss()
        self.hwnd: Optional[int] = None
        self.monitor: Optional[Dict] = None
        
        # 3DS screen dimensions (native resolution)
        self.TOP_SCREEN_HEIGHT = 240
        self.TOP_SCREEN_WIDTH = 400
        self.BOTTOM_SCREEN_HEIGHT = 240
        self.BOTTOM_SCREEN_WIDTH = 320
        
        # Default observation size for RL (can be adjusted)
        self.obs_width = 160
        self.obs_height = 144
        
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
            self._update_monitor_region()
            return True
        return False
    
    def _update_monitor_region(self) -> None:
        """Update the capture region based on window position."""
        if self.hwnd is None:
            return
            
        try:
            rect = win32gui.GetWindowRect(self.hwnd)
            # rect = (left, top, right, bottom)
            self.monitor = {
                "left": rect[0],
                "top": rect[1],
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1]
            }
        except Exception as e:
            print(f"Error getting window rect: {e}")
            self.monitor = None
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture the current frame from the Citra window.
        
        Returns:
            Numpy array of shape (H, W, 3) in BGR format, or None if capture fails.
        """
        if self.hwnd is None:
            if not self.find_window():
                return None
        
        # Update window position in case it moved
        self._update_monitor_region()
        
        if self.monitor is None:
            return None
        
        try:
            screenshot = self.sct.grab(self.monitor)
            frame = np.array(screenshot)
            # Convert BGRA to BGR (remove alpha channel)
            return frame[:, :, :3]
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def capture_top_screen(self) -> Optional[np.ndarray]:
        """
        Capture only the top screen (main gameplay area).
        
        Note: This assumes default Citra layout. You may need to adjust
        the crop coordinates based on your Citra window configuration.
        
        Returns:
            Numpy array of the top screen, or None if capture fails.
        """
        frame = self.capture_frame()
        if frame is None:
            return None
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Estimate top screen region (typically upper portion)
        # This may need adjustment based on Citra settings
        # Default layout: top screen is above bottom screen
        top_screen = frame[:h // 2, :, :]
        
        return top_screen
    
    def get_observation(self, grayscale: bool = False) -> Optional[np.ndarray]:
        """
        Get a processed observation suitable for RL training.
        
        Args:
            grayscale: If True, convert to grayscale.
        
        Returns:
            Resized frame of shape (obs_height, obs_width, channels).
        """
        frame = self.capture_top_screen()
        if frame is None:
            return None
        
        # Resize for neural network input
        resized = cv2.resize(frame, (self.obs_width, self.obs_height))
        
        if grayscale:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            resized = np.expand_dims(resized, axis=-1)
        
        return resized
    
    def set_observation_size(self, width: int, height: int) -> None:
        """Set the output observation size."""
        self.obs_width = width
        self.obs_height = height
    
    def get_window_info(self) -> Dict:
        """
        Get information about the current capture window.
        
        Returns:
            Dictionary with window information.
        """
        if self.hwnd is None:
            return {"found": False}
        
        try:
            title = win32gui.GetWindowText(self.hwnd)
            rect = win32gui.GetWindowRect(self.hwnd)
            return {
                "found": True,
                "hwnd": self.hwnd,
                "title": title,
                "position": {"x": rect[0], "y": rect[1]},
                "size": {"width": rect[2] - rect[0], "height": rect[3] - rect[1]}
            }
        except Exception:
            return {"found": False}


if __name__ == "__main__":
    # Quick test
    import time
    
    print("Testing ScreenCapture module...")
    capture = ScreenCapture()
    
    if capture.find_window():
        info = capture.get_window_info()
        print(f"Found Citra window: {info}")
        
        print("Capturing frames for 3 seconds...")
        start = time.time()
        frame_count = 0
        
        while time.time() - start < 3:
            frame = capture.capture_frame()
            if frame is not None:
                frame_count += 1
        
        fps = frame_count / 3
        print(f"Captured {frame_count} frames ({fps:.1f} FPS)")
        
        # Save a sample frame
        frame = capture.capture_frame()
        if frame is not None:
            cv2.imwrite("test_capture.png", frame)
            print("Saved test_capture.png")
    else:
        print("Could not find Citra window. Make sure Citra is running.")
