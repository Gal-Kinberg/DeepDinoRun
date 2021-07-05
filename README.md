#Deep Dino Run
No internet? No Problem! We will teach a deep neural network to play Chrome's Dino Game!
A pytorch implementation..

Based on the paper:

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou,
Daan Wierstra, Martin Riedmiller [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf), NIPS 2013

![dino run](https://github.com/Gal-Kinberg/DeepDinoRun/blob/revert_checks/dino_run.gif)


- [Deep Dino Run](#Deep Dino Run)
  * [Background](#background)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [References](#references)

## Background
The main goal of this project was to combine the Deep Learning concepts taught in the course with Reinforcement Learning (RL) methods, in order to teach a neural network to play Chrome’s Dino Run game directly from input pixels of the screen. The secondary goal was to compare the performance of different Deep Reinforcement Learning (DRL) architectures and methods on this task: DQN, dueling DQN, and data augmentations.

## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.7`|
|`torch`|  `1.9.0`|
|`selenium`|  `3.141.0`|
|`Pillow`|  `8.3.0`|
|`torchvision`|  `0.10.0`|

## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`DeepDino.py`| main application for training/playing a DQN agent|
|`game.py`| interaction with the game|
|`game_state.py`|holds the agent and the game, returns current state|
|`config.py`| contains paths and urls, hyperparameters, training settings|
|`dino_agent.py`| agent class|
|`dqn_model.py`| DQN classes, neural networks structures|
|`*.pth`| Checkpoint files for the Agents (playing/continual learning)|

## References
* 
* Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou,
Daan Wierstra, Martin Riedmiller [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf), NIPS 2013