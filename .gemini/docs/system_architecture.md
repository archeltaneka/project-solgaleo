## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Loop                            │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │   RL Agent  │◄───│ Reward Signal│◄───│  Game State   │  │
│  │  (Neural    │    │  Calculator  │    │   Extractor   │  │
│  │   Network)  │    └──────────────┘    └───────────────┘  │
│  └──────┬──────┘                               ▲            │
│         │                                      │            │
│         │ Actions                              │            │
│         ▼                                      │            │
│  ┌─────────────────────────────────────────────┴─────────┐  │
│  │              Citra Emulator Interface                 │  │
│  │  - Input injection (keyboard/controller simulation)   │  │
│  │  - Screen capture (game visuals)                      │  │
│  │  - Memory reading (RAM state access)                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                           ▲                                  │
│                           │                                  │
└───────────────────────────┼──────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  Citra Emulator│
                    │  Pokemon Sun/  │
                    │      Moon      │
                    └────────────────┘
```

### Component Breakdown

1. **Citra Emulator Interface**: Bridges Python code with the running emulator
2. **Game State Extractor**: Reads memory and screen to determine current state
3. **RL Agent**: Neural network that decides actions based on observations
4. **Reward Calculator**: Evaluates progress and assigns rewards
5. **Training Loop**: Orchestrates the interaction and learning process