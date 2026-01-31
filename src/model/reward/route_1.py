import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class Route1RewardConfig:
    """Configuration for Route 1 reward shaping"""
    # Core weights
    exploration_weight: float = 1.0      # Visiting new tiles
    progress_weight: float = 2.0         # Moving toward goal direction
    event_weight: float = 5.0            # Battles, items, dialogue triggers
    
    # Penalties
    stuck_penalty: float = -0.5          # Per step when stationary
    menu_penalty: float = -0.1           # Per step in menu/dialogue
    backtrack_penalty: float = -0.2      # Moving away from goal
    
    # Detection thresholds
    movement_threshold: int = 50         # Min pixels moved to count as movement
    stuck_frames: int = 60               # Frames before considered stuck (1 sec at 60fps)
    dialogue_white_min: float = 0.2
    dialogue_white_max: float = 0.8
    dialogue_text_min: float = 0.01
    dialogue_text_max: float = 0.6
    dialogue_edge_threshold: float = 10000.0 
    
    # Route 1 specific: Goal direction vector (Sun/Moon: generally Right-Down)
    goal_direction: Tuple[int, int] = (1, 0)  # (x, y) unit vector


class Route1RewardCalculator:
    """
    Reward calculator for Route 1 exploration training.
    Uses computer vision to detect movement, battles, and progress.
    """
    
    def __init__(self, config: Route1RewardConfig = None):
        self.config = config or Route1RewardConfig()
        
        # State tracking
        self.position_history = deque(maxlen=100)  # Last 100 positions for stuck detection
        self.visited_tiles = set()                  # Spatial coverage (grid-based)
        self.tile_size = 20                         # Grid resolution (20x20px tiles)
        
        # Templates for detection (you'll need to capture these once from Citra)
        self.templates = {
            'battle_indicator': None,    # The "Wild Pokemon appeared!" flash or battle UI
            'dialogue_box': None,        # Bottom text box area
            'trainer_exclamation': None, # Red "!" bubble
            'menu_open': None,           # Bag/Pokemon menu distinctive colors
        }
        
        # Load templates (initialize as None, user captures these)
        self._load_default_templates()
        
        # Previous frame for optical flow
        self.prev_gray = None
        
        # Metrics for logging
        self.stats = {
            'total_distance': 0,
            'unique_tiles_visited': 0,
            'battles_triggered': 0,
            'time_stuck': 0
        }
    
    def _load_default_templates(self):
        """Initialize simple heuristics if no templates provided"""
        # Battle detection: Look for specific color ranges (health bars are red/green)
        self.battle_colors = {
            'lower': np.array([0, 100, 50]),    # HSV lower bound for red/green
            'upper': np.array([80, 255, 255])   # HSV upper bound
        }
    
    def calculate_reward(self, screen: np.ndarray, action_idx: int) -> Tuple[float, Dict]:
        """
        Calculate reward for current frame.
        
        Args:
            screen: 400x240 RGB numpy array from Citra
            action_idx: Action taken (for penalty analysis)
            
        Returns:
            reward: Scalar reward value
            info: Dict with debug info (position, detections, etc.)
        """
        info = {}
        reward = 0.0
        
        # Initialize progress to avoid UnboundLocalError
        progress = 0.0
        
        # 1. Detect game state (Overworld, Battle, Menu, Dialogue)
        game_state = self._detect_game_state(screen)
        info['state'] = game_state
        
        # 2. Calculate movement via Optical Flow
        flow_reward, current_pos = self._analyze_movement(screen)
        reward += flow_reward * self.config.exploration_weight
        
        # 3. Progress toward goal (Route 1 runs SE generally)
        if current_pos and len(self.position_history) > 0:
            prev_pos = self.position_history[-1]
            progress = self._calculate_progress(prev_pos, current_pos)
            reward += progress * self.config.progress_weight
            
            # Backtrack penalty
            if progress < -0.1:
                reward += self.config.backtrack_penalty
        
        # 4. Event detection (battles, trainers spotted)
        event_reward = self._detect_events(screen, game_state)
        reward += event_reward * self.config.event_weight
        
        # 5. Penalties
        penalties = self._calculate_penalties(screen, game_state, action_idx)
        reward += penalties
        
        # 6. Check completion/heuristics
        info.update(self.stats)
        info['reward_components'] = {
            'movement': flow_reward,
            'progress': progress,
            'events': event_reward,
            'penalties': penalties
        }
        
        return reward, info
 
    def _detect_game_state(self, screen: np.ndarray) -> str:
        """
        Classify current game state using proper 3DS screen layout.
        
        Layout:
        - Top 60%: 3DS Top Screen (main gameplay, dialogue appears at bottom of this)
        - Bottom 40%: 3DS Bottom Screen (touch interface)
        """
        h, w = screen.shape[:2]
        
        # Define screen regions (adjust ratios based on your Citra layout)
        top_screen_end = int(h * 0.5)  # Top screen ends around 60% down
        top_screen = screen[:top_screen_end, :]
        bottom_screen = screen[top_screen_end:, :]
        
       # 1. BATTLE (highest priority - check top screen for health bars)
        if self._is_battle_active(top_screen, bottom_screen):
            return 'battle'
        
        # 2. DIALOGUE (check top screen bottom area)
        if self._is_dialogue_open(top_screen):
            return 'dialogue'
        
        # 3. MENU (check bottom screen)
        if self._is_menu_open(bottom_screen):
            return 'menu'
        
        # 4. Default to overworld
        return 'overworld'
    
    def _is_menu_open(self, screen: np.ndarray) -> bool:
        """Detect menu by looking for structured button layout on bottom screen."""
        # Convert to grayscale
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        
        # Look for high contrast edges (buttons have borders)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Menu has moderate edge density (not too sparse like empty, not too dense like noise)
        has_structure = 0.04 < edge_density < 0.30
        
        # Check for grid-like pattern (buttons arranged in grid)
        # Use horizontal and vertical line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        h_lines = np.sum(horizontal_lines > 0)
        v_lines = np.sum(vertical_lines > 0)
        
        # Menu should have both horizontal and vertical structure
        has_grid = h_lines > 100 and v_lines > 100
        
        return has_structure and has_grid

    def _is_dialogue_open(self, top_screen: np.ndarray) -> bool:
        """
        Detect dialogue box on the bottom portion of the top screen.
        
        Sun/Moon dialogue characteristics:
        - White/light background box at bottom of top screen
        - Black text inside
        - Often has character portrait on left side
        - Distinctive rounded shape
        - Sometimes has "next page" arrow indicator
        """
        h, w = top_screen.shape[:2]
        
        # Focus on bottom 30% of top screen where dialogue appears
        dialogue_region = top_screen[int(h * 0.6):, :]
        
        if dialogue_region.size == 0:
            return False
        
        # Convert to grayscale
        gray = cv2.cvtColor(dialogue_region, cv2.COLOR_RGB2GRAY)
        
        # Method 1: Look for high white area (dialogue background) with dark text
        white_mask = gray > 200  # Very bright pixels (dialogue box background)
        white_ratio = np.sum(white_mask) / white_mask.size
        
        # Dialogue box typically covers 30-70% of the width and 40-80% of height
        has_white_box = 0.1 < white_ratio < 0.8
        
        # Method 2: Look for black text (dark pixels on white background)
        dark_mask = gray < 80  # Dark text
        dark_ratio = np.sum(dark_mask) / dark_mask.size
        
        # Should have some text but not too much
        has_text = 0.01 < dark_ratio < 0.6
        
        # Method 3: Check for horizontal line (top border of dialogue box)
        # The dialogue box has a distinctive top edge
        top_edge = gray[5:15, :]  # Check near top of dialogue region
        edge_variance = np.var(top_edge)
        has_edge = edge_variance > 5000  # High variance = edge detected
        
        # Combine detection methods
        # Require white box + text, portrait is bonus but not required
        is_dialogue = has_white_box and has_text and has_edge
        
        # cv2.imshow('Dialogue Region', dialogue_region)
        # print(f"Dialogue detection: white={white_ratio:.2f}, text={dark_ratio:.2f}, edge={edge_variance:.2f}")
        
        return is_dialogue

    def _is_battle_active(self, top_screen: np.ndarray, bottom_screen: np.ndarray) -> bool:
        """
        Detect if we're in a Pokemon battle.
        
        Battle characteristics:
        - Top screen: Health bars (red/yellow/green), Pokemon sprites, level numbers
        - Bottom screen: Fight/Bag/Pokemon/Run menu OR move selection
        - Distinctive UI layout different from overworld/menu
        """
        # Method 1: Health bars on top screen (most reliable)
        if self._detect_health_bars(top_screen):
            print("Health bars detected")
            return True
        
        # Method 2: Battle menu detection on bottom screen
        if self._is_battle_menu(bottom_screen):
            print("Battle menu detected")
            return True
        
        # Method 3: Battle transition effects (flash, etc)
        if self._detect_battle_transition(top_screen):
            print("Battle transition detected")
            return True
            
        return False
    
    def _detect_health_bars(self, top_screen: np.ndarray) -> bool:
        """
        Detect health bars on top screen.
        Sun/Moon layout:
        - Opponent (enemy): Top RIGHT corner
        - Player: Bottom LEFT corner
        """
        h, w = top_screen.shape[:2]
        
        # Enemy health bar: Top right region
        enemy_region = top_screen[:int(h*0.25), int(w*0.55):]
        
        # Player health bar: Bottom left region  
        player_region = top_screen[int(h*0.80):int(h*0.95), :int(w*0.40)]

        # Convert to HSV for color detection
        hsv_enemy = cv2.cvtColor(enemy_region, cv2.COLOR_RGB2HSV)
        hsv_player = cv2.cvtColor(player_region, cv2.COLOR_RGB2HSV)
        
        # Health bar colors
        # Red: Low HP or border color
        lower_red1 = np.array([0, 100, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Green/Teal: High HP (Sun/Moon uses slightly teal-ish green)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])
        
        # Yellow: Medium HP
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        # Detect colors in both regions
        red_mask_enemy = cv2.inRange(hsv_enemy, lower_red1, upper_red1) | \
                        cv2.inRange(hsv_enemy, lower_red2, upper_red2)
        red_mask_player = cv2.inRange(hsv_player, lower_red1, upper_red1) | \
                         cv2.inRange(hsv_player, lower_red2, upper_red2)
        
        green_mask_enemy = cv2.inRange(hsv_enemy, lower_green, upper_green)
        green_mask_player = cv2.inRange(hsv_player, lower_green, upper_green)
        
        yellow_mask_enemy = cv2.inRange(hsv_enemy, lower_yellow, upper_yellow)
        yellow_mask_player = cv2.inRange(hsv_player, lower_yellow, upper_yellow)
        
        # Count pixels
        enemy_hp_pixels = np.sum(red_mask_enemy > 0) + np.sum(green_mask_enemy > 0) + np.sum(yellow_mask_enemy > 0)
        player_hp_pixels = np.sum(red_mask_player > 0) + np.sum(green_mask_player > 0) + np.sum(yellow_mask_player > 0)
        
        # Debug visualization (optional)
        # cv2.imshow('Enemy HP Region', enemy_region)
        # cv2.imshow('Player HP Region', player_region)
        # cv2.waitKey(0)
        # print(f"Enemy HP pixels: {enemy_hp_pixels}, Player HP pixels: {player_hp_pixels}")
        
        # Battle detected if we see health indicators in either location
        # Both present = definite battle, one present = likely battle
        has_enemy_hp = enemy_hp_pixels > 100
        has_player_hp = player_hp_pixels > 150  # Player bar is usually larger
        
        return has_enemy_hp or has_player_hp
    
    def _is_battle_menu(self, bottom_screen: np.ndarray) -> bool:
        """
        Detect Sun/Moon battle menu with rounded colorful buttons.
        
        Distinctive features:
        - Cyan/teal background (battle arena floor)
        - Red/Orange "FIGHT" button (right side)
        - Green "POKÉMON" button (top left)
        - Yellow/Gold "BAG" button (middle left)  
        - Blue "RUN" button (bottom)
        - Much more saturated than X-menu
        """
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(bottom_screen, cv2.COLOR_RGB2HSV)
        
        # 1. Detect distinctive battle menu colors
        # Red/Orange (FIGHT button) - High saturation, warm hue
        lower_red_orange = np.array([0, 120, 100])
        upper_red_orange = np.array([25, 255, 255])
        fight_mask = cv2.inRange(hsv, lower_red_orange, upper_red_orange)
        fight_pixels = np.sum(fight_mask > 0)
        
        # Green (POKÉMON button)
        lower_green = np.array([35, 100, 50])
        upper_green = np.array([85, 255, 200])
        pokemon_mask = cv2.inRange(hsv, lower_green, upper_green)
        pokemon_pixels = np.sum(pokemon_mask > 0)
        
        # Yellow/Gold (BAG button)
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        bag_mask = cv2.inRange(hsv, lower_yellow, lower_yellow)
        bag_pixels = np.sum(bag_mask > 0)
        
        # Blue (RUN button)
        lower_blue = np.array([90, 80, 50])
        upper_blue = np.array([130, 255, 255])
        run_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        run_pixels = np.sum(run_mask > 0)
        
        # Cyan/Teal background (battle floor)
        lower_cyan = np.array([0, 191, 165])
        upper_cyan = np.array([64, 240, 224])
        bg_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
        bg_pixels = np.sum(bg_mask > 0)
        
        # Debug output (uncomment to tune)
        # print(f"Battle colors: FIGHT={fight_pixels}, PKMN={pokemon_pixels}, "
        #       f"BAG={bag_pixels}, RUN={run_pixels}, BG={bg_pixels}")
        
        # 2. Battle signature detection
        # Must have cyan/teal background (distinguishes from X-menu)
        has_cyan_bg = bg_pixels > (bottom_screen.size * 0.01)
        
        # Must have multiple button colors present (at least 2 of the 4)
        button_colors_detected = sum([
            fight_pixels > 500,
            pokemon_pixels > 300,
            bag_pixels > 300,
            run_pixels > 500
        ])
        has_multiple_buttons = button_colors_detected >= 2
        
        # 3. Texture check: Battle menu is busier than X-menu
        # Has lots of edges due to button borders/icons
        gray = cv2.cvtColor(bottom_screen, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # X-menu: ~2-6% edges, Battle menu: ~8-15% edges (more complex UI)
        is_complex_ui = 0.06 < edge_density < 0.20
        
        return has_cyan_bg and has_multiple_buttons and is_complex_ui
    
    def _detect_battle_transition(self, top_screen: np.ndarray) -> bool:
        """
        Detect battle start/end transitions.
        - Flash effects (screen brightens suddenly)
        - Vignette effect (darkened corners during intro)
        """
        # Check for extreme brightness (flash)
        mean_brightness = np.mean(top_screen)
        
        # Battle flash is very bright (240+) or very dark vignette (< 40)
        is_flash = mean_brightness > 150 or mean_brightness < 30
        
        return is_flash
    
    def _analyze_movement(self, screen: np.ndarray) -> Tuple[float, Tuple[int, int]]:
        """
        Use optical flow to detect actual character movement.
        Returns reward based on new tiles visited.
        """
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, None
        
        # Calculate dense optical flow using Farneback (robust, no tracking points needed)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # flow is shape (H, W, 2) where [:,:,0] is x flow, [:,:,1] is y flow
        fx, fy = flow[:,:,0], flow[:,:,1]
        
        # Calculate magnitude to filter noise
        mag, _ = cv2.cartToPolar(fx, fy)
        
        # Only consider significant movement (filter out noise/static)
        significant = mag > 0.5
        
        if not np.any(significant):
            # No significant movement detected
            self.prev_gray = gray
            return 0.0, (screen.shape[1]//2, screen.shape[0]//2)
        
        # Calculate mean flow direction from significant movements
        mean_dx = np.mean(fx[significant])
        mean_dy = np.mean(fy[significant])
        mean_movement = np.sqrt(mean_dx**2 + mean_dy**2)
        
        # Update position estimate (simplified: use center + cumulative flow)
        h, w = screen.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Determine grid tile based on current view
        tile_x = center_x // self.tile_size
        tile_y = center_y // self.tile_size
        tile_key = (tile_x, tile_y)
        
        # Exploration reward
        if tile_key not in self.visited_tiles:
            self.visited_tiles.add(tile_key)
            self.stats['unique_tiles_visited'] = len(self.visited_tiles)
            exploration_bonus = 1.0  # New tile!
        else:
            exploration_bonus = 0.1  # Revisiting known area
        
        # Update stats
        self.stats['total_distance'] += mean_movement
        
        # Store current frame for next iteration
        self.prev_gray = gray
        
        return exploration_bonus, (center_x, center_y)
    
    def _calculate_progress(self, prev_pos: Tuple[int, int], 
                          curr_pos: Tuple[int, int]) -> float:
        """
        Calculate progress toward goal direction.
        Route 1 generally flows from NW to SE (or similar).
        Adjust goal_direction based on your specific checkpoint location.
        """
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        
        # Dot product with goal direction
        goal_x, goal_y = self.config.goal_direction
        progress = (dx * goal_x + dy * goal_y) / (np.sqrt(dx**2 + dy**2 + 1e-6))
        
        # Normalize to reasonable range [-1, 2] (positive is good)
        return progress * 0.5
    
    def _detect_events(self, screen: np.ndarray, game_state: str) -> float:
        """Detect significant game events"""
        reward = 0.0
        
        # Battle start detection (transition to battle state)
        if game_state == 'battle' and hasattr(self, '_last_state'):
            if self._last_state != 'battle':
                reward += 1.0  # Entered battle
                self.stats['battles_triggered'] += 1
        
        # Trainer detection (red exclamation mark appearing)
        if self._detect_trainer_spotted(screen):
            reward += 0.5  # Found a trainer
        
        self._last_state = game_state
        return reward
    
    def _detect_trainer_spotted(self, screen: np.ndarray) -> bool:
        """Detect the '!' bubble when spotted by trainer"""
        # Convert to HSV and look for red (high saturation, specific hue)
        hsv = cv2.cvtColor(screen, cv2.COLOR_RGB2HSV)
        # Red appears at both ends of HSV hue spectrum
        mask1 = cv2.inRange(hsv, np.array([0, 150, 150]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 150, 150]), np.array([180, 255, 255]))
        red_mask = mask1 | mask2
        
        # Look for circular blob (exclamation mark)
        red_pixels = np.sum(red_mask > 0)
        return red_pixels > 100  # Threshold
    
    def _calculate_penalties(self, screen: np.ndarray, game_state: str, 
                           action_idx: int) -> float:
        """Calculate penalties for undesirable behavior"""
        penalty = 0.0
        
        # Stuck detection (no movement for N frames)
        if len(self.position_history) >= self.config.stuck_frames:
            recent_positions = list(self.position_history)[-self.config.stuck_frames:]
            # Calculate variance in position
            xs = [p[0] for p in recent_positions]
            ys = [p[1] for p in recent_positions]
            variance = np.var(xs) + np.var(ys)
            
            if variance < 10:  # Not moving
                penalty += self.config.stuck_penalty
                self.stats['time_stuck'] += 1
        
        # Menu spam penalty (mashing buttons in menus doesn't help)
        if game_state in ['menu', 'dialogue']:
            penalty += self.config.menu_penalty
        
        # Invalid action penalty (e.g., pressing UP during dialogue doesn't advance)
        if game_state == 'dialogue' and action_idx in [8, 9, 10, 11]:  # D-pad
            penalty -= 0.05  # Small nudge to use A/B instead
        
        return penalty
    
    def is_done(self, screen: np.ndarray, step_count: int) -> bool:
        """
        Check if episode should terminate:
        - Reached Pokemon Center (checkpoint 2)
        - Stuck for too long (2000+ frames)
        - Defeated in battle (blackout)
        """
        # Timeout
        if step_count > 5000:  # ~2-3 minutes at 30fps
            return True
        
        # Check for blackout (all black screen with specific text)
        if self._detect_blackout(screen):
            return True
        
        # Check if reached Pokemon Center (visual template match)
        # You'd need to capture the "Pokemon Center" entrance sprite
        # if self._detect_at_pokemon_center(screen):
        #     return True  # Success!
        
        return False
    
    def _detect_blackout(self, screen: np.ndarray) -> bool:
        """Detect whiteout/blackout (defeat)"""
        gray = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        return np.mean(gray) < 20 or np.mean(gray) > 240
    
    def reset(self):
        """Call at start of new episode"""
        self.position_history.clear()
        self.visited_tiles.clear()
        self.prev_gray = None
        self.stats = {k: 0 for k in self.stats}
        self._last_state = None