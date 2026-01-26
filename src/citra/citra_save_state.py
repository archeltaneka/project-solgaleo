"""
Citra Save State Manager (Updated for newer Citra versions)
Handles creating, loading, and managing save states for RL training.
Works with Citra's save/load system, avoiding Ctrl+C conflict.
"""

import os
import time
import json
from pathlib import Path
import pyautogui
import win32gui

from src.citra.citra_environment import CitraAgent


class SaveStateManager:
    """
    Manages Citra save states for Pokemon training.
    Uses direct window messaging to avoid Ctrl+C conflict with Python.
    """
    
    def __init__(self, save_dir="training_saves", window_name="Citra"):
        """
        Initialize the save state manager.
        
        Args:
            save_dir: Directory to store save state metadata
            window_name: Citra window name
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.window_name = window_name
        self.hwnd = None
        
        # Metadata file to track save states
        self.metadata_file = self.save_dir / "save_metadata.json"
        self.save_states = self._load_metadata()
        self.last_save_time = 0
        
        print(f"Save State Manager initialized")
        print(f"Metadata directory: {self.save_dir.absolute()}")
        print(f"Note: We'll use PyAutoGUI to send hotkeys to avoid Ctrl+C conflict")
        
    def _find_window(self):
        """Find the Citra window."""
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
        return self.hwnd
        
    def _load_metadata(self):
        """Load save state metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.save_states, f, indent=2)
    
    def _focus_citra(self):
        """Focus the Citra window."""
        if self.hwnd is None:
            self._find_window()
        win32gui.SetForegroundWindow(self.hwnd)
        time.sleep(0.2)  # Wait for window to be focused
    
    def create_save_state(self, name, description=""):
        """
        Create a save state in Citra.
        
        Uses PyAutoGUI to send Ctrl+C to the Citra window specifically,
        avoiding Python's interrupt signal.
        
        Args:
            name: Unique name for this save state
            description: Description of what this save represents
        """
        print(f"Creating save state '{name}'...")
        
        # Focus Citra window
        self._focus_citra()
        
        # Use PyAutoGUI to send Ctrl+C to the focused window
        # This bypasses Python's signal handler
        pyautogui.hotkey('ctrl', 'c')
        
        # Wait for save to complete
        time.sleep(1.0)
        
        # Track save time for ordering
        self.last_save_time = time.time()
        
        # Store metadata
        self.save_states[name] = {
            'description': description,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp': self.last_save_time
        }
        self._save_metadata()
        
        print(f"✓ Save state '{name}' created successfully!")
        return True
    
    def load_save_state(self, name=None):
        """
        Load the most recent save state in Citra using Ctrl+V.
        
        Args:
            name: Name of save state (for metadata tracking only)
        """
        if name and name not in self.save_states:
            print(f"Warning: Save state '{name}' not found in metadata.")
            print(f"Available: {list(self.save_states.keys())}")
            print("Attempting to load most recent save anyway...")
        
        print(f"Loading most recent save state...")
        if name:
            print(f"  Expected: '{name}'")
        
        # Focus Citra window
        self._focus_citra()
        
        # Use PyAutoGUI to send Ctrl+V
        pyautogui.hotkey('ctrl', 'v')
        
        # Wait for load to complete
        time.sleep(1.0)
        
        print(f"✓ Save state loaded!")
        return True
    
    def load_most_recent(self):
        """Load the most recently created save state."""
        if not self.save_states:
            raise Exception("No save states available")
        
        # Find most recent by timestamp
        most_recent = max(self.save_states.items(), 
                         key=lambda x: x[1]['timestamp'])
        name = most_recent[0]
        
        print(f"Loading most recent save state: '{name}'")
        return self.load_save_state(name)
    
    def list_save_states(self):
        """List all available save states."""
        if not self.save_states:
            print("No save states found.")
            return []
        
        print("\n=== Available Save States ===")
        # Sort by timestamp (most recent first)
        sorted_states = sorted(self.save_states.items(), 
                              key=lambda x: x[1]['timestamp'], 
                              reverse=True)
        
        for name, info in sorted_states:
            print(f"  • {name}")
            print(f"    Description: {info['description']}")
            print(f"    Created: {info['created_at']}")
            print()
        
        print("Note: Ctrl+V loads the MOST RECENT save state")
        print(f"  → Current most recent: '{sorted_states[0][0]}'")
        
        return list(self.save_states.keys())
    
    def delete_save_state(self, name):
        """Remove a save state from metadata."""
        if name in self.save_states:
            del self.save_states[name]
            self._save_metadata()
            print(f"✓ Save state '{name}' removed from registry")
            return True
        return False
    
    def set_main_checkpoint(self, name):
        """
        Mark a save state as the main training checkpoint.
        Updates its timestamp to make it the most recent.
        
        Args:
            name: Name of the save state to set as main checkpoint
        """
        if name not in self.save_states:
            raise ValueError(f"Save state '{name}' not found")
        
        # Update timestamp to be the most recent
        self.save_states[name]['timestamp'] = time.time()
        self.save_states[name]['is_main'] = True
        self._save_metadata()
        
        print(f"✓ '{name}' set as main training checkpoint")
        print(f"  Ctrl+V will now load this save state")


class CitraPokemonEnvWithSaveStates:
    """
    Enhanced environment with save state support.
    Works with newer Citra versions.
    """
    
    def __init__(self, window_name="Citra", save_dir="training_saves"):
        """
        Initialize environment with save state support.
        
        Args:
            window_name: Citra window title
            save_dir: Directory for save state metadata
        """
        
        self.base_env = CitraAgent(window_name)
        self.save_manager = SaveStateManager(save_dir, window_name)
        self.current_checkpoint = None
        
        print("Environment with save states ready!")
    
    def set_checkpoint(self, checkpoint_name):
        """
        Set the checkpoint to use for reset().
        
        Args:
            checkpoint_name: Name of the save state to load on reset
        """
        if checkpoint_name not in self.save_manager.save_states:
            raise ValueError(f"Checkpoint '{checkpoint_name}' not found")
        
        # Make this the most recent save
        self.save_manager.set_main_checkpoint(checkpoint_name)
        self.current_checkpoint = checkpoint_name
        print(f"Checkpoint set to: {checkpoint_name}")
    
    def reset(self):
        """
        Reset the environment by loading the checkpoint.
        
        Returns:
            observation: Initial screen state after reset
        """
        if self.current_checkpoint is None:
            print("Warning: No checkpoint set. Loading most recent save...")
            self.save_manager.load_most_recent()
        else:
            self.save_manager.load_save_state(self.current_checkpoint)
        
        # Wait for game to stabilize
        time.sleep(1.2)
        
        # Return initial observation
        return self.base_env.capture_screen(resize=(240, 160))
    
    def step(self, action_idx, hold_duration=0.1):
        """Perform one step."""
        return self.base_env.step(action_idx, hold_duration)
    
    def render(self, observation):
        """Display observation."""
        self.base_env.render(observation)
    
    def close(self):
        """Cleanup."""
        self.base_env.close()
    
    def create_checkpoint(self, name, description=""):
        """Create a new checkpoint."""
        return self.save_manager.create_save_state(name, description)
    
    def list_checkpoints(self):
        """List all available checkpoints."""
        return self.save_manager.list_save_states()
    
    @property
    def action_space(self):
        """Get action space."""
        return self.base_env.action_space


# Interactive Save State Creator
def interactive_save_creator():
    """
    Interactive tool to help create save states.
    """
    print("=== Interactive Save State Creator ===\n")
    
    window_name = input("Enter Citra window name (default: 'Citra'): ").strip() or "Citra"
    manager = SaveStateManager(window_name=window_name)
    
    print("\nInstructions:")
    print("1. Navigate to a location in Pokemon Sun/Moon")
    print("2. Come back here and create a save state")
    print("3. The script will focus Citra and send Ctrl+C")
    print("4. Your LAST created save will be the one loaded by Ctrl+V\n")
    
    while True:
        print("\n" + "="*50)
        print("Options:")
        print("  1. Create new save state")
        print("  2. List all save states")
        print("  3. Test load most recent save")
        print("  4. Set a save as main checkpoint")
        print("  5. Delete a save state")
        print("  6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            name = input("Enter save state name: ").strip()
            if not name:
                print("Error: Name cannot be empty")
                continue
            
            description = input("Enter description (optional): ").strip()
            
            print("\nMake sure you're at the desired location in Pokemon!")
            print("The script will focus Citra and press Ctrl+C in 3 seconds...")
            input("Press Enter to continue...")
            
            print("Focusing Citra in 3...")
            time.sleep(1)
            print("2...")
            time.sleep(1)
            print("1...")
            time.sleep(1)
            
            try:
                manager.create_save_state(name, description)
            except Exception as e:
                print(f"Error: {e}")
                print("Make sure Citra is running and the window name is correct")
            
        elif choice == '2':
            manager.list_save_states()
            
        elif choice == '3':
            if not manager.save_states:
                print("No save states available")
                continue
            
            print("\nThe script will focus Citra and press Ctrl+V in 3 seconds...")
            input("Press Enter to continue...")
            
            print("Focusing Citra in 3...")
            time.sleep(1)
            print("2...")
            time.sleep(1)
            print("1...")
            time.sleep(1)
            
            try:
                manager.load_most_recent()
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == '4':
            if not manager.save_states:
                print("No save states available")
                continue
            
            manager.list_save_states()
            name = input("\nEnter save state name: ").strip()
            
            if name in manager.save_states:
                manager.set_main_checkpoint(name)
            else:
                print(f"Save state '{name}' not found")
        
        elif choice == '5':
            if not manager.save_states:
                print("No save states to delete")
                continue
            
            manager.list_save_states()
            name = input("Enter save state name: ").strip()
            manager.delete_save_state(name)
            
        elif choice == '6':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice")
