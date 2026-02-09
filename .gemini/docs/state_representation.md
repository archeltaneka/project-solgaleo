## State Representation

### Observation Space Design

For Pokemon, the state should include both visual and numerical information:

**1. Visual State (CNN Input):**
```python
# Top screen: 400x240 RGB or grayscale
# Can be downscaled for efficiency: 84x84 or 160x120
visual_state_shape = (84, 84, 3)  # or (84, 84, 1) for grayscale
```

**2. Memory-Based State (Dense Features):**
```python
game_state = {
    # Player info
    'player_position': (x, y, map_id),  # 3D coordinates
    'player_direction': 0-3,             # Facing direction
    'money': int,
    'badges': int,
    
    # Party Pokemon
    'party': [
        {
            'species_id': int,
            'level': int,
            'hp_current': int,
            'hp_max': int,
            'status': int,  # 0=normal, 1=poisoned, etc.
            'moves': [move_id1, move_id2, move_id3, move_id4]
        }
        # ... up to 6 Pokemon
    ],
    
    # Battle state (if in battle)
    'in_battle': bool,
    'opponent_pokemon': {...},
    'can_run': bool,
    
    # Progress indicators
    'pokedex_seen': int,
    'pokedex_caught': int,
    'story_flags': [bool] * 1000,  # Important story progression flags
    
    # Menu/UI state
    'in_menu': bool,
    'dialog_open': bool,
    'current_menu_selection': int
}
```

**3. Hybrid Approach (Recommended):**
```python
class PokemonObservation:
    def __init__(self):
        # Visual: Processed screen (for CNN)
        self.screen = np.zeros((84, 84, 3), dtype=np.uint8)
        
        # Numeric: Game state vector (for MLP)
        self.vector_state = np.zeros(256, dtype=np.float32)
        # Includes: position, party stats, flags, etc.
    
    def get_combined_state(self):
        """Returns both visual and vector observations"""
        return {
            'screen': self.screen,
            'vector': self.vector_state
        }
```