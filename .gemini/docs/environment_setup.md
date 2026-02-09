## Environment Setup

### Prerequisites

**Required Software:**
- Python 3.8+ (recommended: 3.10)
- Citra Emulator (Nightly or Canary build for best compatibility)
- Pokemon Sun or Moon ROM (legally obtained)
- Windows 10/11 64-bit

**Python Libraries:**
```bash
# Core ML/RL frameworks
pip install torch torchvision  # PyTorch for neural networks
pip install stable-baselines3  # RL algorithms
pip install gymnasium          # Environment interface (OpenAI Gym successor)

# Emulator interface
pip install pyautogui          # For input simulation
pip install mss                # Fast screen capture
pip install opencv-python      # Image processing
pip install numpy pandas       # Data manipulation

# Memory reading
pip install pymem              # Windows memory reading
pip install psutil             # Process management

# Utilities
pip install pillow             # Image handling
pip install matplotlib         # Visualization
pip install tensorboard        # Training monitoring
```

### Citra Configuration

**Emulator Settings for AI Training:**

1. **Graphics Settings:**
   - Resolution: 1x Native (400x240) or 2x for better state capture
   - Use hardware renderer
   - Disable VSync for faster training
   - Disable frame limiting initially

2. **General Settings:**
   - Enable "Use custom user directory" for save state management
   - Configure hotkeys for save states programmatically

3. **Performance:**
   - Fast-forward hotkey configured for accelerated training
   - Limit frame rate to consistent value (e.g., 60 FPS or uncapped)