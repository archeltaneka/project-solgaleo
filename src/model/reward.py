"""
Pokemon Sun/Moon Reward System for RL Training
Provides reward signals based on visual observation and game progress.
"""

import numpy as np
import cv2
from collections import deque


class PokemonRewardCalculator:
    """
    Calculates rewards for Pokemon Sun/Moon RL agent.
    Uses vision-based heuristics since we don't have memory reading.
    """
    
    def __init__(self, checkpoint_type="exploration"):
        """
        Initialize reward calculator.
        
        Args:
            checkpoint_type: Type of training scenario
                - "exploration": Reward for exploring new areas
                - "battle": Reward for battle performance
                - "catching": Reward for catching Pokemon
                - "progression": Reward for story progress
        """
        self.checkpoint_type = checkpoint_type
        
        # Track history for detecting changes
        self.position_history = deque(maxlen=20)
        self.screen_history = deque(maxlen=10)
        self.visited_areas = set()
        
        # Counters
        self.steps_without_progress = 0
        self.total_steps = 0
        
        print(f"Reward Calculator initialized for: {checkpoint_type}")
    
    def calculate_reward(self, prev_obs, curr_obs, action):
        """
        Calculate reward based on observation changes.
        
        Args:
            prev_obs: Previous screen observation (numpy array)
            curr_obs: Current screen observation (numpy array)
            action: Action taken (from action_space)
            
        Returns:
            float: Reward value
        """
        reward = 0.0
        self.total_steps += 1
        
        # Base reward structure depends on checkpoint type
        if self.checkpoint_type == "exploration":
            reward = self._exploration_reward(prev_obs, curr_obs, action)
        elif self.checkpoint_type == "battle":
            reward = self._battle_reward(prev_obs, curr_obs, action)
        elif self.checkpoint_type == "catching":
            reward = self._catching_reward(prev_obs, curr_obs, action)
        elif self.checkpoint_type == "progression":
            reward = self._progression_reward(prev_obs, curr_obs, action)
        
        # Universal penalties
        reward += self._universal_penalties(prev_obs, curr_obs, action)
        
        return reward
    
    def _exploration_reward(self, prev_obs, curr_obs, action):
        """Reward for exploring new areas (Route 1 training)."""
        reward = 0.0
        
        # 1. Reward for screen movement (character is moving)
        movement_detected = self._detect_movement(prev_obs, curr_obs)
        if movement_detected:
            reward += 0.1
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1
        
        # 2. Reward for visiting new screen areas
        # Only count as new area if NOT in a menu
        if not self._is_menu_screen(curr_obs):
            area_hash = self._hash_screen_area(curr_obs)
            is_new_area = True
            
            # Check if this area is similar to any previously visited area
            for visited_hash in self.visited_areas:
                # Calculate Hamming distance (how many bits differ)
                hamming_dist = bin(area_hash ^ visited_hash).count('1')
                similarity = 1.0 - (hamming_dist / 256.0)  # 16x16 = 256 bits
                
                # If similarity > 85%, consider it the same area
                if similarity > 0.85:
                    is_new_area = False
                    break
            
            if is_new_area:
                self.visited_areas.add(area_hash)
                reward += 1.0  # Big reward for new areas
                print(f"  [Reward] New area/cutscene discovered! +1.0 (Total areas: {len(self.visited_areas)})")
        
        # 3. Penalty for getting stuck
        if self.steps_without_progress > 50:
            reward -= 0.5
        
        # 4. Small reward for forward movement (directional bias)
        if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
            reward += 0.01
        
        # 5. Penalty for opening menus during exploration (we want movement, not menu browsing)
        if self._is_menu_screen(curr_obs):
            print(f"  [Penalty] Menu opened! -0.2")
            reward -= 0.2  # Small penalty for opening menu
        
        return reward
    
    def _battle_reward(self, prev_obs, curr_obs, action):
        """Reward for battle performance."""
        reward = 0.0
        
        # 1. Detect HP changes (damage dealt/received)
        hp_change = self._detect_hp_change(prev_obs, curr_obs)
        
        if hp_change < 0:
            # We took damage
            reward -= 0.3
        elif hp_change > 0:
            # We dealt damage (enemy HP decreased)
            reward += 0.5
        
        # 2. Detect battle victory (screen changes significantly)
        if self._detect_battle_end(prev_obs, curr_obs):
            reward += 5.0  # Large reward for winning
            print(f"  [Reward] Battle won! +5.0")
        
        # 3. Reward for selecting moves (not just spamming A)
        if action in ['A', 'B']:
            reward += 0.05
        
        # 4. Penalty for fainting (screen goes dark/white)
        if self._detect_faint(curr_obs):
            reward -= 10.0
            print(f"  [Reward] Pokemon fainted! -10.0")
        
        return reward
    
    def _catching_reward(self, prev_obs, curr_obs, action):
        """Reward for catching Pokemon."""
        reward = 0.0
        
        # 1. Detect Pokeball throw (screen shake/animation)
        if self._detect_pokeball_throw(prev_obs, curr_obs):
            reward += 0.5
        
        # 2. Detect successful catch (Pokedex entry, sound, etc.)
        if self._detect_catch_success(prev_obs, curr_obs):
            reward += 10.0
            print(f"  [Reward] Pokemon caught! +10.0")
        
        # 3. Detect catch failure (Pokemon broke free)
        if self._detect_catch_failure(prev_obs, curr_obs):
            reward -= 1.0
        
        return reward
    
    def _progression_reward(self, prev_obs, curr_obs, action):
        """Reward for story progression."""
        reward = 0.0
        
        # 1. Detect dialogue/cutscene progression
        if self._detect_dialogue(curr_obs):
            if action == 'A':  # Advancing dialogue
                reward += 0.2
        
        # 2. Detect reaching new locations (town entrance, etc.)
        if self._detect_new_location(prev_obs, curr_obs):
            reward += 3.0
            print(f"  [Reward] New location reached! +3.0")
        
        # 3. Detect trial completion
        if self._detect_trial_completion(prev_obs, curr_obs):
            reward += 20.0
            print(f"  [Reward] Trial completed! +20.0")
        
        return reward
    
    def _universal_penalties(self, prev_obs, curr_obs, action):
        """Penalties that apply to all scenarios."""
        penalty = 0.0
        
        # 1. Time penalty (encourage efficiency)
        penalty -= 0.01
        
        # 2. Penalty for repeating same action too much
        if len(self.position_history) > 10:
            recent_actions = list(self.position_history)[-10:]
            if len(set(recent_actions)) == 1:  # Same action 10 times
                penalty -= 0.2
        
        # 3. Penalty for NONE action (doing nothing)
        if action == 'NONE':
            penalty -= 0.05
        
        return penalty
    
    # ===== Detection Helper Functions =====
    
    def _detect_movement(self, prev_obs, curr_obs):
        """Detect if the screen changed (character moved)."""
        if prev_obs is None or curr_obs is None:
            return False
        
        # Calculate difference between frames
        diff = cv2.absdiff(prev_obs, curr_obs)
        diff_score = np.mean(diff)
        
        # If difference is significant, movement occurred
        return diff_score > 5.0
    
    def _hash_screen_area(self, obs):
        """
        Create a robust hash of the current screen area for novelty detection.
        Uses perceptual hashing to handle camera movement and minor pixel changes.
        """
        if obs is None:
            return 0
        
        # Downsample significantly to remove fine details
        small = cv2.resize(obs, (16, 16))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        
        # Apply thresholding to reduce sensitivity to lighting/camera shifts
        # Convert to binary: dark pixels = 0, bright pixels = 1
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Create perceptual hash using average hash algorithm
        # This is much more robust to minor changes
        avg = np.mean(binary)
        hash_bits = (binary > avg).astype(int)
        
        # Convert to a single hash value
        # Areas that look similar will have similar hash values
        hash_value = 0
        for i, bit in enumerate(hash_bits.flatten()):
            if bit:
                hash_value |= (1 << i)
        
        return hash_value
    
    def _detect_hp_change(self, prev_obs, curr_obs):
        """
        Detect HP bar changes (simplified - looks at top screen area).
        Returns positive if enemy HP decreased, negative if ours decreased.
        """
        if prev_obs is None or curr_obs is None:
            return 0
        
        # HP bars are usually in top portion of screen
        # This is a simplified heuristic - would need tuning
        prev_top = prev_obs[:40, :, :]
        curr_top = curr_obs[:40, :, :]
        
        # Detect color changes (HP bars are often green/yellow/red)
        diff = np.mean(cv2.absdiff(prev_top, curr_top))
        
        # Simplified: if top area changed significantly, assume HP change
        if diff > 10:
            return np.random.choice([-1, 1])  # Placeholder
        return 0
    
    def _detect_battle_end(self, prev_obs, curr_obs):
        """Detect if battle ended (significant screen change)."""
        if prev_obs is None or curr_obs is None:
            return False
        
        diff = np.mean(cv2.absdiff(prev_obs, curr_obs))
        # Large diff suggests transition out of battle
        return diff > 30
    
    def _detect_faint(self, obs):
        """Detect if Pokemon fainted (screen goes dark/white)."""
        if obs is None:
            return False
        
        # Check if screen is very dark or very bright
        brightness = np.mean(obs)
        return brightness < 20 or brightness > 240
    
    def _detect_pokeball_throw(self, prev_obs, curr_obs):
        """Detect Pokeball throw animation."""
        # Simplified - would need specific pattern matching
        return False
    
    def _detect_catch_success(self, prev_obs, curr_obs):
        """Detect successful Pokemon catch."""
        # Would look for specific UI elements
        return False
    
    def _detect_catch_failure(self, prev_obs, curr_obs):
        """Detect Pokemon breaking free."""
        return False
    
    def _detect_dialogue(self, obs):
        """Detect dialogue box on screen."""
        if obs is None:
            return False
        
        # Dialogue boxes are usually in bottom portion
        bottom = obs[-60:, :, :]
        
        # Check for white/dark box pattern
        # This is simplified - would need better detection
        return np.mean(bottom) > 100
    
    def _detect_new_location(self, prev_obs, curr_obs):
        """Detect entering a new location (screen transition)."""
        if prev_obs is None or curr_obs is None:
            return False
        
        diff = np.mean(cv2.absdiff(prev_obs, curr_obs))
        return diff > 50  # Very large change suggests location change
    
    def _detect_trial_completion(self, prev_obs, curr_obs):
        """Detect trial completion (would need specific pattern)."""
        # Would look for Z-Crystal acquisition screen, specific text, etc.
        return False
    
    def _is_menu_screen(self, obs):
        """
        Detect if current screen is a menu (bag, Pokédex, party, etc.).
        Menus typically have:
        - Lots of UI elements (boxes, text)
        - High contrast edges
        - Specific color patterns
        - Static backgrounds
        """
        if obs is None:
            return False
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        
        # Detect edges (menus have lots of UI boxes/borders)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Menus typically have high edge density (lots of UI elements)
        # Normal gameplay has lower edge density (more natural scenery)
        if edge_density > 0.15:  # Threshold for "lots of edges"
            return True
        
        # Additional check: Look for large uniform color blocks
        # Menus often have solid color backgrounds/panels
        # Sample a few regions and check if they're very uniform
        h, w = obs.shape[:2]
        
        # Check center region (menus usually have UI in center)
        center_region = obs[h//4:3*h//4, w//4:3*w//4, :]
        std_dev = np.std(center_region)
        
        # Low standard deviation = uniform color = likely a menu panel
        if std_dev < 30:
            return True
        
        # Check for high saturation (menu buttons are often colorful)
        hsv = cv2.cvtColor(obs, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        high_sat_ratio = np.sum(saturation > 100) / saturation.size
        
        # Menus often have colorful buttons/icons
        if high_sat_ratio > 0.3 and edge_density > 0.10:
            return True
        
        return False
    
    def reset(self):
        """Reset reward calculator for new episode."""
        self.position_history.clear()
        self.screen_history.clear()
        self.visited_areas.clear()
        self.steps_without_progress = 0
        self.total_steps = 0


# Example usage in training loop
class RewardWrapper:
    """
    Wrapper to integrate reward calculation into environment.
    """
    
    def __init__(self, env, checkpoint_type="exploration"):
        """
        Wrap environment with reward calculation.
        
        Args:
            env: CitraPokemonEnv instance
            checkpoint_type: Training scenario type
        """
        self.env = env
        self.reward_calc = PokemonRewardCalculator(checkpoint_type)
        self.prev_obs = None
        
    def reset(self):
        """Reset environment and reward calculator."""
        self.reward_calc.reset()
        obs = self.env.reset()
        self.prev_obs = obs
        return obs
    
    def step(self, action_idx):
        """
        Take a step and calculate reward.
        
        Args:
            action_idx: Index of action in action_space
            
        Returns:
            observation, reward, done, info
        """
        action_name = self.env.action_space[action_idx]
        
        # Execute action
        curr_obs = self.env.step(action_idx)
        
        # Calculate reward
        reward = self.reward_calc.calculate_reward(
            self.prev_obs, 
            curr_obs, 
            action_name
        )
        
        # Check if episode should end
        done = self._check_done()
        
        # Additional info
        info = {
            'steps': self.reward_calc.total_steps,
            'visited_areas': len(self.reward_calc.visited_areas),
            'stuck_steps': self.reward_calc.steps_without_progress
        }
        
        self.prev_obs = curr_obs
        
        return curr_obs, reward, done, info
    
    def _check_done(self):
        """Determine if episode should end."""
        # Episode ends if:
        # 1. Max steps reached
        if self.reward_calc.total_steps >= 1000:
            return True
        
        # 2. Stuck for too long
        if self.reward_calc.steps_without_progress > 100:
            return True
        
        # 3. Goal reached (would need specific detection)
        # For now, just use step limit
        
        return False
    
    def render(self, observation):
        """Display observation."""
        self.env.render(observation)
    
    def close(self):
        """Cleanup."""
        self.env.close()
    
    @property
    def action_space(self):
        """Get action space."""
        return self.env.action_space
