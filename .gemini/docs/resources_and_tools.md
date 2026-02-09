## Resources & Tools

### Essential Tools

1. **Citra Emulator**
   - Download: https://citra-emu.org/
   - Nightly builds recommended

2. **Cheat Engine**
   - Download: https://www.cheatengine.org/
   - For finding memory addresses

3. **Python Libraries**
   ```bash
   pip install stable-baselines3 torch gymnasium
   pip install pymem pyautogui mss opencv-python
   pip install tensorboard matplotlib numpy pandas
   ```

4. **PKHeX** (Optional)
   - Pokemon save editor
   - Useful for understanding data structures
   - Download: https://projectpokemon.org/home/files/file/1-pkhex/

### Learning Resources

**Reinforcement Learning:**
- Stable-Baselines3 Docs: https://stable-baselines3.readthedocs.io/
- Spinning Up in RL: https://spinningup.openai.com/
- Sutton & Barto Book: http://incompleteideas.net/book/

**Pokemon Reverse Engineering:**
- Project Pokemon: https://projectpokemon.org/
- Pokemon ROM hacking communities
- Bulbapedia (game mechanics): https://bulbapedia.bulbagarden.net/

**Related Projects:**
- PokemonRed-Bot: https://github.com/topics/pokemon-bot
- OpenAI Gym Retro: https://github.com/openai/retro

### Community & Support

- r/MachineLearning (Reddit)
- r/reinforcementlearning (Reddit)
- Stable-Baselines3 Discord
- Pokemon ROM Hacking Discord servers

---

## Performance Expectations

### Training Timeline

**Hardware Requirements:**
- CPU: Intel i5/AMD Ryzen 5 or better
- RAM: 16GB minimum, 32GB recommended
- GPU: NVIDIA GTX 1060 or better (for neural network training)
- Storage: 50GB free space (for models, logs, save states)

**Training Time Estimates:**
- Tutorial completion: 1-2 days
- First gym/trial: 1 week
- Mid-game proficiency: 2-3 weeks
- Full game completion: 1-2 months (highly variable)

**Note:** These are rough estimates. Actual performance depends on:
- Reward function quality
- Hyperparameter tuning
- Curriculum design
- Hardware capabilities
- Algorithm selection

### Success Metrics

Track these metrics to evaluate progress:

```python
metrics = {
    'episode_reward': [],
    'episode_length': [],
    'battles_won': [],
    'badges_obtained': [],
    'pokemon_caught': [],
    'map_coverage': [],  # % of map explored
    'story_completion': [],  # % of story flags triggered
    'training_time': []
}
```