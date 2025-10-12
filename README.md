# COMS4061A/7071A – Project: Reinforcement Learning in the Crafter Environment

This project explores **reinforcement learning (RL)** in open-world survival environments using the **Crafter** benchmark. The goal is to design and train agents capable of **surviving and thriving** in a procedurally generated 2D world, learning to manage hunger and health, collect resources, craft tools, and unlock achievements through exploration and combat.  

The project focuses on:
- Implementing **two RL algorithms**:
  - One **covered** in the course: `Deep Q Network`
  - One **not covered** in the course: `Proximal Policy Optimisation`
- Iteratively **improving both agents** based on experimental evaluation.
- Evaluating and comparing their **performance, efficiency, and generalisation**.

## Group members
- **Sahil Maharaj** - 2550404@students.wits.ac.za
- **Karabo Mohapeloa** - 2998002@students.wits.ac.za
- **Kgetja Bruce Mphekgwane** - 2593733@students.wits.ac.za 
- **Ntando Raji** - 2584925@students.wits.ac.za

## Environment: Crafter  

**Crafter** is a 2D pixel-based, procedurally generated survival game designed for benchmarking general RL agents.  

- **Observation space:** 64×64 RGB image  
- **Action space:** 17 discrete actions  
- **Rewards:**  
  - +1 per timestep survived  
  - +1 for each achievement (22 total achievements)  
- **Mechanics:** Hunger, health, crafting, combat, and resource collection  

**Resources:**
- Crafter GitHub: [https://github.com/danijar/crafter](https://github.com/danijar/crafter)  
- Crafter Paper: [https://arxiv.org/pdf/2109.06780](https://arxiv.org/pdf/2109.06780)


## Project Structure 
```
COMS4061A_Project_Crafter/
│
├── scripts/
|   ├── utilities                # Utility functions for evaluation, etc.
│   ├── dqn/                     # DQN agent implementation and training
│   └── ppo/                     # PPO agent implementation and training
|       ├── baseline/            # Baseline PPO agent
|       |   ├── train_model.py    # Train baseline agent
|       |   └── evaluate_model.py # Evaluate baseline agent     
|       ├── env/                 # Environment alterations specific to the PPO agents
|       └── models/              # Save PPO agent models    
├── logs/                        # Training logs and TensorBoard summaries
├── videos/                      # Rendered agent videos
├── results/                     # Evaluation metrics and graphs
├── environment.yml              # Environment dependencies
└── README.md                    # Project overview and setup instructions
```

## Getting Started
### 1️. Clone the Repository  
```bash
git clone 
cd COMS4061A_Project_Crafter
```

### 2️. Create and Activate the Environment  
Use the provided YAML file for version compatibility.  
```bash
conda env create -f environment.yml
conda activate crafter_env
```

### 3. Install Crafter and Dependencies  
```bash
pip install stable-baselines3 opencv-python
pip install git+https://github.com/catid/crafter.git
```

### 4. Train an Agent  
```bash
python scripts/ppo/baseline/train_agent.py

usage: train_agent.py [-h] [--outdir OUTDIR] [--steps STEPS] [--model_path MODEL_PATH] [--video_path VIDEO_PATH] [--log_dir LOG_DIR] [--results_dir RESULTS_DIR]

options:
  -h, --help            show help message and exit
  --outdir OUTDIR
  --steps STEPS
  --model_path MODEL_PATH
  --video_path VIDEO_PATH
  --log_dir LOG_DIR
  --results_dir RESULTS_DIR
```

### 5️. Evaluate and Render Results
```bash
python scripts/ppo/baseline/evaluate_agent.py

usage: evaluate_agent.py [-h] --model_path MODEL_PATH [--video_path VIDEO_PATH] [--num_episodes NUM_EPISODES]

options:
  -h, --help            show help message and exit
  --model_path MODEL_PATH
                        Path to trained PPO model (.zip)
  --video_path VIDEO_PATH
                        Optional path to save video (e.g., ./eval_run.mp4)
  --num_episodes NUM_EPISODES
                        Number of evaluation episodes
```

### 6. Visualize Training Metrics  
```bash
tensorboard --logdir logs/
```

## Project Workflow  

### Base Phase
- Implement and evaluate a baseline **DQN agent** (from course material).
- Implement and evaluate a **PPO agent** (not covered in the course). 

## Acknowledgements  
- [Danijar Hafner](https://github.com/danijar) for the Crafter environment.  
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) developers. 
- Crafter issue fix reference: https://github.com/danijar/crafter/issues/8
