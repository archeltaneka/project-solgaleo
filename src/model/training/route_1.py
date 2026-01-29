from src.model.reward import Route1RewardCalculator
from src.model.env import CitraPokemonEnvWithSaveStates


class Route1TrainingEnv(CitraPokemonEnvWithSaveStates):
    def __init__(self, checkpoint_name="route1_start"):
        super().__init__()
        self.set_checkpoint(checkpoint_name)
        self.reward_calc = Route1RewardCalculator()
        self.step_count = 0
        self.episode_reward = 0
        
    def step(self, action_idx):
        observation = super().step(action_idx)
        reward, info = self.reward_calc.calculate_reward(observation, action_idx)
        
        self.step_count += 1
        self.episode_reward += reward
        
        done = self.reward_calc.is_done(observation, self.step_count)
        
        if done:
            info['episode_reward'] = self.episode_reward
            info['final_stats'] = self.reward_calc.stats
            
        return observation, reward, done, info
    
    def reset(self):
        obs = super().reset()
        self.reward_calc.reset()
        self.step_count = 0
        self.episode_reward = 0
        return obs