# Snake Game with Deep Q-Learning (DQL)

This project implements the classic Snake game using Deep Q-Learning (DQL) to train an AI agent to play the game intelligently.

## Project Overview

The purpose of this project is to apply reinforcement learning techniques — specifically Deep Q-Learning — to teach an agent how to play Snake. The agent observes the environment, makes decisions based on a learned policy, and improves over time by receiving rewards.

We experimented with **two different approaches** to training the agent:

1. **Simple Q-Learning with basic state representation:**  
   - Faster to train, but limited by the lack of spatial awareness and depth in state encoding.

2. **Deep Q-Learning using neural networks and full state-space representation:**  
   - Utilizes a neural network (PyTorch) to approximate Q-values, allowing more complex and generalized learning.

## Why Two Methods?

By comparing a basic tabular Q-learning approach with a deep learning-based approach, we aim to understand:
- How neural networks improve policy generalization
- Performance differences in training time and stability
- Practical limitations of using simple vs. deep methods in reinforcement learning

## Libraries and Dependencies

This project uses the following Python libraries:

- `numpy`: for numerical operations
- `matplotlib`: for visualizations
- `torch` (PyTorch): for building and training the neural network
- `pygame`: for rendering the Snake game environment
- `random` and `collections`: for managing game state and experience replay

You can install the dependencies with:

```bash
pip install -r requirements.txt
