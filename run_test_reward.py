"""
main.py - Manual Testing Script for Route 1 Reward System
Run this while manually controlling Citra to validate reward calculations.
"""

import cv2
import numpy as np
import time
import json
from datetime import datetime
from collections import deque
import argparse

from src.citra.citra_save_state import CitraPokemonEnvWithSaveStates
from src.citra.citra_environment import CitraPokemonEnv  # Fallback if needed


class RewardVisualizer:
    """
    Real-time visualization overlay for the reward system.
    Draws debug info on top of the captured screen.
    """
    
    def __init__(self, scale=1.5):
        self.scale = scale
        self.colors = {
            'state_overworld': (0, 255, 0),    # Green
            'state_battle': (0, 0, 255),       # Red
            'state_dialogue': (255, 255, 0),   # Yellow
            'state_menu': (255, 0, 255),       # Purple
            'movement': (0, 255, 255),         # Cyan
            'grid': (50, 50, 50),              # Dark gray
            'text': (255, 255, 255),           # White
            'alert': (0, 165, 255)             # Orange
        }
        
    def draw_overlay(self, screen, reward_info, total_reward, fps, calibration_mode=False):
        """
        Draw visualization overlay on the screen.
        
        Args:
            screen: 400x240 RGB image
            reward_info: Dict from reward calculator
            total_reward: Cumulative reward so far
            fps: Current FPS
            calibration_mode: If True, show calibration helpers
        """
        img = screen.copy()
        h, w = img.shape[:2]
        
        # Resize if needed for better visibility
        if self.scale != 1.0:
            img = cv2.resize(img, (int(w*self.scale), int(h*self.scale)), 
                           interpolation=cv2.INTER_NEAREST)
            scale_factor = self.scale
        else:
            scale_factor = 1.0
        
        # 1. Draw Header Background
        header_h = 80
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], header_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # 2. State Indicator (Large colored pill)
        state = reward_info.get('state', 'unknown')
        state_colors = {
            'overworld': self.colors['state_overworld'],
            'battle': self.colors['state_battle'],
            'dialogue': self.colors['state_dialogue'],
            'menu': self.colors['state_menu']
        }
        state_color = state_colors.get(state, (128, 128, 128))
        
        cv2.rectangle(img, (10, 10), (150, 40), state_color, -1)
        cv2.rectangle(img, (10, 10), (150, 40), (255, 255, 255), 2)
        cv2.putText(img, f"STATE: {state.upper()}", (20, 32), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 3. Reward Display
        components = reward_info.get('reward_components', {})
        current_reward = sum(components.values()) if components else 0
        
        # Main reward counter (top right)
        reward_text = f"Total: {total_reward:+.2f}"
        reward_color = (0, 255, 0) if current_reward >= 0 else (0, 0, 255)
        cv2.putText(img, reward_text, (img.shape[1]-200, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, reward_color, 2)
        
        # Instant reward (smaller, below)
        instant_text = f"Frame: {current_reward:+.2f}"
        cv2.putText(img, instant_text, (img.shape[1]-200, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # 4. Component Breakdown (Left side panel)
        y_offset = header_h + 20
        if components:
            cv2.putText(img, "REWARD COMPONENTS:", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
            for comp_name, val in components.items():
                color = (0, 255, 0) if val >= 0 else (0, 0, 255)
                text = f"{comp_name}: {val:+.2f}"
                cv2.putText(img, text, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
        
        # 5. Stats (Right side panel)
        stats = reward_info.get('final_stats', reward_info)
        x_stats = img.shape[1] - 180
        y_stats = header_h + 20
        
        cv2.putText(img, "STATS:", (x_stats, y_stats), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_stats += 20
        
        stat_items = [
            f"Tiles: {stats.get('unique_tiles_visited', 0)}",
            f"Distance: {stats.get('total_distance', 0):.0f}px",
            f"Battles: {stats.get('battles_triggered', 0)}",
            f"Stuck: {stats.get('time_stuck', 0)}s",
            f"FPS: {fps:.1f}"
        ]
        
        for stat in stat_items:
            cv2.putText(img, stat, (x_stats, y_stats), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_stats += 20
        
        # 6. Movement Visualization (Grid overlay)
        if calibration_mode:
            # Draw grid
            tile_size = int(20 * scale_factor)
            for x in range(0, img.shape[1], tile_size):
                cv2.line(img, (x, header_h), (x, img.shape[0]), self.colors['grid'], 1)
            for y in range(header_h, img.shape[0], tile_size):
                cv2.line(img, (0, y), (img.shape[1], y), self.colors['grid'], 1)
            
            # Show center point
            cx, cy = img.shape[1]//2, (img.shape[0]+header_h)//2
            cv2.circle(img, (cx, cy), 5, self.colors['movement'], -1)
            cv2.putText(img, "CENTER", (cx+10, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['movement'], 1)
        
        # 7. Alerts/Warnings
        if stats.get('time_stuck', 0) > 5:
            cv2.rectangle(img, (img.shape[1]//2-100, img.shape[0]//2-30), 
                         (img.shape[1]//2+100, img.shape[0]//2+10), (0, 0, 0), -1)
            cv2.putText(img, "STUCK DETECTED!", (img.shape[1]//2-90, img.shape[0]//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['alert'], 2)
        
        if state == 'battle':
            cv2.rectangle(img, (img.shape[1]//2-80, 100), 
                         (img.shape[1]//2+80, 140), (0, 0, 0), -1)
            cv2.putText(img, "BATTLE MODE", (img.shape[1]//2-70, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['state_battle'], 2)
        
        return img


class ManualTester:
    """
    Main testing harness for manual Citra control.
    """
    
    def __init__(self, checkpoint="route1_start", save_dir="training_saves"):
        print("="*50)
        print("Route 1 Reward System - Manual Test Harness")
        print("="*50)
        print("\nInitializing...")
        
        # Import here to avoid circular dependencies
        from src.model.reward.route_1 import Route1RewardCalculator, Route1RewardConfig
        
        self.env = CitraPokemonEnvWithSaveStates(
            window_name="Citra",
            save_dir=save_dir
        )
        
        # Load checkpoint
        try:
            self.env.set_checkpoint(checkpoint)
            print(f"✓ Loaded checkpoint: {checkpoint}")
        except ValueError as e:
            print(f"⚠ Checkpoint error: {e}")
            print("Continuing in observation mode without checkpoint...")
        
        # Reward system
        config = Route1RewardConfig(
            exploration_weight=1.0,
            progress_weight=2.0,
            event_weight=5.0,
            stuck_penalty=-0.5
        )
        self.reward_calc = Route1RewardCalculator(config)
        self.visualizer = RewardVisualizer(scale=1.5)
        
        # Session tracking
        self.total_reward = 0.0
        self.episode_rewards = []
        self.frame_count = 0
        self.start_time = None
        self.calibration_mode = False
        
        # Logging
        self.log_data = []
        self.log_file = f"reward_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        print("\nControls:")
        print("  ESC - Quit")
        print("  R - Reset reward calculator")
        print("  P - Pause")
        print("  C - Toggle calibration mode (shows grid)")
        print("  S - Save current frame as template")
        print("  L - Load checkpoint (reset to start)")
        print("\nMake sure Citra is running and visible!")
        input("Press Enter when ready to start...")
        
    def run(self):
        """Main observation loop."""
        print("\nStarting observation...")
        self.start_time = time.time()
        last_time = self.start_time
        fps = 0
        
        # Initial observation
        try:
            obs = self.env.reset()
        except Exception as e:
            print(f"Warning: Could not reset environment: {e}")
            print("Attempting to capture current screen...")
            obs = self.env.base_env.capture_screen(resize=(400, 240))
        
        self.reward_calc.reset()
        
        running = True
        paused = False
        
        while running:
            if not paused:
                # Capture screen
                try:
                    obs = self.env.base_env.capture_screen(resize=(400, 240))
                except Exception as e:
                    print(f"Screen capture failed: {e}")
                    break
                
                # Calculate reward (action_idx=-1 indicates manual control)
                reward, info = self.reward_calc.calculate_reward(obs, action_idx=-1)
                self.total_reward += reward
                self.frame_count += 1
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - last_time + 1e-6)
                last_time = current_time
                
                # Log data every 30 frames (~1 second)
                if self.frame_count % 30 == 0:
                    log_entry = {
                        'frame': self.frame_count,
                        'time': current_time - self.start_time,
                        'total_reward': self.total_reward,
                        'state': info.get('state'),
                        'stats': {k: v for k, v in info.items() if k != 'reward_components'}
                    }
                    self.log_data.append(log_entry)
                
                # Create visualization
                viz_img = self.visualizer.draw_overlay(
                    obs, info, self.total_reward, fps, 
                    calibration_mode=self.calibration_mode
                )
                
                # Display
                cv2.imshow('Route 1 Reward Test', viz_img)
            
            # Handle keyboard input (with small delay for responsiveness)
            key = cv2.waitKey(30) & 0xFF
            
            if key == 27:  # ESC
                running = False
            elif key == ord('r'):
                print("\n[RESET] Reward calculator reset")
                self.reward_calc.reset()
                self.total_reward = 0.0
                self.frame_count = 0
                self.start_time = time.time()
            elif key == ord('p'):
                paused = not paused
                print(f"\n[{'PAUSED' if paused else 'RESUMED'}]")
            elif key == ord('c'):
                self.calibration_mode = not self.calibration_mode
                print(f"\n[CALIBRATION MODE: {'ON' if self.calibration_mode else 'OFF'}]")
            elif key == ord('s'):
                # Save current frame as template
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"template_capture_{timestamp}.png"
                cv2.imwrite(filename, cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
                print(f"\n[SCREENSHOT] Saved as {filename}")
            elif key == ord('l'):
                print("\n[LOAD] Resetting to checkpoint...")
                try:
                    obs = self.env.reset()
                    self.reward_calc.reset()
                    self.total_reward = 0.0
                except Exception as e:
                    print(f"Failed to load: {e}")
        
        # Cleanup
        self._save_logs()
        cv2.destroyAllWindows()
        self.env.close()
        
        print("\n" + "="*50)
        print("Testing session ended")
        print(f"Final Reward: {self.total_reward:.2f}")
        print(f"Frames captured: {self.frame_count}")
        print(f"Duration: {time.time() - self.start_time:.1f}s")
        print(f"Log saved: {self.log_file}")
        print("="*50)
        
    def _save_logs(self):
        """Save session logs to file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump({
                    'session_info': {
                        'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                        'total_frames': self.frame_count,
                        'final_reward': self.total_reward,
                        'checkpoint': 'route1_start'
                    },
                    'frame_data': self.log_data
                }, f, indent=2)
        except Exception as e:
            print(f"Failed to save logs: {e}")


def quick_capture_mode():
    """
    Simple mode: Just capture and show screen with basic FPS counter.
    Useful for testing if the environment works at all.
    """
    print("Quick Capture Mode - Press ESC to quit")
    env = CitraPokemonEnv()
    
    while True:
        start = time.time()
        screen = env.capture_screen(resize=(400, 240))
        
        # Simple FPS counter
        fps = 1.0 / (time.time() - start + 1e-6)
        screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
        cv2.putText(screen_bgr, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Citra Capture Test', screen_bgr)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()


def calibration_walkthrough():
    """
    Interactive calibration to set the correct goal direction for Route 1.
    """
    print("="*50)
    print("ROUTE 1 CALIBRATION WALKTHROUGH")
    print("="*50)
    print("\nThis will help you determine the correct goal direction vector.")
    print("1. Load your Route 1 start checkpoint")
    print("2. Walk straight toward the Pokemon Center")
    print("3. Observe the X,Y changes in the debug display")
    print("4. Use those values to set goal_direction in the config")
    print("\nPress any key when ready...")
    input()
    
    from src.model.reward.route_1 import Route1RewardCalculator
    
    env = CitraPokemonEnv()
    calc = Route1RewardCalculator()
    calc.position_history.maxlen = 10  # Shorter history for quick response
    
    print("\nTracking movement... Walk toward your goal!")
    print("Watch the 'Progress' value - positive is good, negative is wrong direction")
    
    positions = []
    
    while True:
        screen = env.capture_screen(resize=(400, 240))
        
        # Quick position detection
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        if calc.prev_gray is not None:
            # Simple center of mass detection
            flow = cv2.calcOpticalFlowPyrLK(calc.prev_gray, gray, None, None)
            if flow is not None:
                h, w = screen.shape[:2]
                cx, cy = w//2, h//2
                positions.append((cx, cy))
                
                if len(positions) > 1:
                    dx = positions[-1][0] - positions[-2][0]
                    dy = positions[-1][1] - positions[-2][1]
                    
                    # Draw direction vector
                    screen_bgr = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                    start_pt = (w//2, h//2)
                    end_pt = (w//2 + dx*5, h//2 + dy*5)  # Scale up for visibility
                    
                    cv2.arrowedLine(screen_bgr, start_pt, end_pt, (0, 255, 0), 2)
                    cv2.putText(screen_bgr, f"DX: {dx}, DY: {dy}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(screen_bgr, f"Vector: ({dx}, {dy})", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(screen_bgr, "Goal direction should match this", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.imshow('Calibration', screen_bgr)
        
        calc.prev_gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        
        if cv2.waitKey(30) & 0xFF == 27:
            break
    
    cv2.destroyAllWindows()
    print("\nCalibration ended.")
    if len(positions) > 5:
        avg_dx = np.mean([positions[i+1][0] - positions[i][0] for i in range(len(positions)-1)])
        avg_dy = np.mean([positions[i+1][1] - positions[i][1] for i in range(len(positions)-1)])
        print(f"Average movement vector: ({avg_dx:.1f}, {avg_dy:.1f})")
        print(f"Suggested goal_direction: ({np.sign(avg_dx)}, {np.sign(avg_dy)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Route 1 Reward System')
    parser.add_argument('--mode', choices=['test', 'quick', 'calibrate'], 
                       default='test', help='Testing mode')
    parser.add_argument('--checkpoint', default='route1_start', 
                       help='Checkpoint name to load')
    parser.add_argument('--scale', type=float, default=1.5, 
                       help='Display scale factor')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_capture_mode()
    elif args.mode == 'calibrate':
        calibration_walkthrough()
    else:
        tester = ManualTester(checkpoint=args.checkpoint)
        tester.run()