## Action Space

### Discrete Action Space

**Basic Action Set (11 actions):**
```python
ACTIONS = {
    0: 'NOOP',        # No operation
    1: 'A',           # Confirm/Interact
    2: 'B',           # Cancel/Run
    3: 'UP',          # Move up
    4: 'DOWN',        # Move down
    5: 'LEFT',        # Move left
    6: 'RIGHT',       # Move right
    7: 'START',       # Menu
    8: 'SELECT',      # Special functions
    9: 'L',           # Left shoulder
    10: 'R'           # Right shoulder
}
```

**Extended Action Set (for complex scenarios):**
```python
# Can include combinations like:
'UP_A',    # Move up while holding A
'DOWN_B',  # Move down while holding B
# etc.
```

### Action Space Implementation

```python
import gymnasium as gym
from gymnasium import spaces

class PokemonEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Define action space
        self.action_space = spaces.Discrete(11)
        
        # Define observation space (hybrid)
        self.observation_space = spaces.Dict({
            'screen': spaces.Box(
                low=0, high=255, 
                shape=(84, 84, 3), 
                dtype=np.uint8
            ),
            'vector': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(256,),
                dtype=np.float32
            )
        })
```