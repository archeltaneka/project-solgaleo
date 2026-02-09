## Reward Function Design

### Multi-Objective Reward System

For general Pokemon gameplay, combine multiple reward signals:

```python
class RewardCalculator:
    def __init__(self):
        self.prev_state = None
        self.weights = {
            'exploration': 1.0,
            'story_progress': 10.0,
            'battle_win': 5.0,
            'pokemon_catch': 3.0,
            'level_up': 2.0,
            'damage_dealt': 0.1,
            'damage_taken': -0.1,
            'pokemon_fainted': -2.0,
            'time_penalty': -0.01,
            'new_location': 2.0,
            'pokedex_entry': 1.0
        }
    
    def calculate_reward(self, prev_state, current_state):
        """Calculate total reward based on state transition"""
        reward = 0.0
        
        # 1. Exploration reward (visited new tiles)
        if self.is_new_location(current_state):
            reward += self.weights['exploration']
        
        # 2. Story progress (event flags, badges)
        if current_state['badges'] > prev_state['badges']:
            reward += self.weights['story_progress']
        
        # 3. Battle outcomes
        if self.battle_ended(prev_state, current_state):
            if self.battle_won(current_state):
                reward += self.weights['battle_win']
        
        # 4. Pokemon catching
        if current_state['pokedex_caught'] > prev_state['pokedex_caught']:
            reward += self.weights['pokemon_catch']
        
        # 5. Level ups
        reward += self.count_level_ups(prev_state, current_state) * \
                  self.weights['level_up']
        
        # 6. Battle performance
        if current_state['in_battle']:
            damage_delta = self.calculate_damage_delta(prev_state, current_state)
            reward += damage_delta['dealt'] * self.weights['damage_dealt']
            reward += damage_delta['taken'] * self.weights['damage_taken']
        
        # 7. Pokemon fainting penalty
        if self.pokemon_fainted(prev_state, current_state):
            reward += self.weights['pokemon_fainted']
        
        # 8. Time penalty (encourage efficiency)
        reward += self.weights['time_penalty']
        
        # 9. New location discovery
        if current_state['map_id'] != prev_state['map_id']:
            reward += self.weights['new_location']
        
        # 10. Pokedex progress
        if current_state['pokedex_seen'] > prev_state['pokedex_seen']:
            reward += self.weights['pokedex_entry']
        
        return reward
    
    def is_new_location(self, state):
        """Check if player is in a previously unvisited location"""
        # Requires maintaining a visited locations map
        pass
    
    def battle_won(self, state):
        """Determine if battle was won"""
        # Check battle result flags
        pass
```

### Reward Shaping Strategies

**Curriculum Learning Rewards:**
```python
class CurriculumRewards:
    """Adjust rewards based on training stage"""
    def __init__(self):
        self.stage = 0  # 0=early game, 1=mid game, 2=late game
    
    def get_stage_weights(self):
        if self.stage == 0:
            # Early: Emphasize exploration and basic mechanics
            return {
                'exploration': 2.0,
                'battle_win': 3.0,
                'story_progress': 10.0
            }
        elif self.stage == 1:
            # Mid: Balance everything
            return {
                'exploration': 1.0,
                'battle_win': 5.0,
                'pokemon_catch': 5.0
            }
        else:
            # Late: Emphasize completion and optimization
            return {
                'story_progress': 15.0,
                'pokedex_entry': 3.0
            }
```