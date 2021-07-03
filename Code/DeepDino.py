# Selenium Imports
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

# PyTorch Imports
import torch
from torch import nn
from torchvision import models

# Python Imports
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from collections import deque

# Extra Imports
from io import BytesIO
import base64
import pickle
import json
import cv2
from PIL import Image
from datetime import datetime

# Import config for paths and scripts
from Code.config import *
from Code.utils.dqn_model import DQN, DuelingDQN
from Code.utils.game import Game
from Code.utils.dino_agent import DinoAgent
from Code.utils.game_state import GameState


def show_img(graphs=False):
    """
    Show images in new window
    """
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        imS = cv2.resize(screen, (800, 400))
        cv2.imshow(window_title, screen)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            cv2.destroyAllWindows()
            break


def set_parameter_requires_grad(model, feature_extracting=False):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0  # image size, e.g. (3, 224, 224)

    if model_name == "resnet":
        """ Resnet50
        """
        model_ft = ResNetDQN(ACTIONS)
        input_size = 224

    elif model_name == "vgg":
        """ VGG16
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "dqn":
        model_ft = DQN(num_classes)
        input_size = 80

    elif model_name == "dueling dqn":
        model_ft = DuelingDQN(num_classes)
        input_size = 80

    elif model_name == "dueling resnet":
        model_ft = DuelingResent(num_classes)
        input_size = 224
        set_parameter_requires_grad(model_ft.conv_layer, feature_extracting=True)

    else:
        raise NotImplementedError

    return model_ft, input_size


### Main Training Function
def train_network(model, model_name, game_state, device, criterion, optimizer, observe=False):
    last_time = time.time()

    # load checkpoint
    if CHECKPOINT:  # TODO: make a more generic loading (check if file exists etc.)
        checkpoint = torch.load(
            f'/content/drive/MyDrive/Deep_Learning/DeepDinoRun/checkpoints/dqn_2000_games_24_06_2021_10-18.pth')
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epsilon = checkpoint['epsilon']
        D = checkpoint['experience']  # experience memory
        t_total = checkpoint['t_total']
        game_state.games_num = checkpoint['games num']
        scores = checkpoint['scores']
        iterations = checkpoint['iterations']

        # TODO: load schedulers
    else:
        D = deque()  # TODO: Change Replay Memory to enable working with dataloaders?
        epsilon = INITIAL_EPSILON
        t_total = 0
        scores = []
        iterations = []

    # TODO: Add data augmentations
    # initialise: perform first action and get initial state
    do_nothing = torch.zeros(ACTIONS)
    do_nothing[0] = 1

    # get initial state
    x_t, r_0, terminal, score = game_state.get_state(do_nothing)

    # stack the first image 4 times as placeholder
    s_t = torch.stack((x_t, x_t, x_t, x_t), dim=2)  # 80x80x4
    s_t = s_t.permute(2, 0, 1)  # change the axis such that the channels is the first dimension: 4x80x80
    # reshape to add a fourth dimension
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1x4x80x80
    initial_state = s_t

    if observe:  # Load current weights and observe
        OBSERVE = 999999999
        epsilon = FINAL_EPSILON

    else:
        OBSERVE = OBSERVATION

    t = 0
    tic_framerate = time.time()
    while True:
        model.eval()
        running_loss = 0.0
        Q_next_state = 0
        chosen_action = 0  # TODO: Change this to enable same action over 4 frames?
        r_t = 0
        a_t = torch.zeros(ACTIONS)  # TODO: Change this to enable same action over 4 frames?
        # chose an epsilon greedy action
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                # print("-----Random Action-----")
                chosen_action = random.randrange(ACTIONS)
            else:
                q = model(s_t.to(device, dtype=torch.float32))
                chosen_action = torch.argmax(q)
            a_t[chosen_action] = 1

        # reduce epsilon
        if epsilon > FINAL_EPSILON and t_total > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the chosen action and observe next state and reward
        x_next, r_t, terminal, score = game_state.get_state(a_t)
        x_next = x_next.reshape(1, 1, x_next.shape[0], x_next.shape[1])  # 1x1x80x80
        s_next = torch.cat((x_next, s_t[:, :3, :, :]), dim=1)  # 1x4x80x80

        # save experience to replay memory, check memory length
        D.append((s_t, chosen_action, r_t, s_next, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # update the state s_t, if the state was terminal, restart the game
        s_t = initial_state if terminal else s_next
        if terminal:
            toc_framerate = time.time()
            last_time = iterations[-1] if len(iterations) > 0 else 0
            framerate = (t_total - last_time) / (toc_framerate - tic_framerate)
            scores.append(score)  # save current game score
            iterations.append(t_total)  # save current number of iterations
            print(
                f' |||| Game: {game_state.games_num} || Score: {score} || Max Score: {np.max(scores)} || Mean of last 10 games: {np.mean(scores[np.max([0, game_state.games_num - 10]):]):.2f} || Epsilon: {epsilon:.2f} || Time: {t_total} || Avarage Framerate: {framerate:.2f} ||||')
            # print(f'Max Score: {np.max(scores)}')
            # print(f'Mean of last 10 games: {np.mean(scores[np.max([0, game_state.games_num-10]):]):.2f}')
            # print(f'Epsilon: {epsilon:.2f}')
            # print("Time = ", t_total)
            # print(f'Avarage Framerate: {framerate:.2f}')
            # time.sleep(3)
            tic_framerate = time.time()

        # update t
        t += 1
        if terminal and game_state.games_num % CHECKPOINT_TIME == 0:
            now = datetime.now()
            date_time = now.strftime("%d_%m_%Y_%H-%M")
            print("Saving Model...")
            state = {
                'net': model.state_dict(),
                'games num': game_state.games_num,
                'optimizer': optimizer.state_dict(),
                't_total': t_total,
                'scores': scores,
                'iterations': iterations,
                'experience': D,
                'epsilon': epsilon,
                'learning rate': LEARNING_RATE,
                'batch size': BATCH,
                'initial epsilon': INITIAL_EPSILON,
                'final epsilon': FINAL_EPSILON,
                'EXPLORE': EXPLORE,
                'TRAIN': TRAIN,
                'OBSERVATION': OBSERVATION,
                'frames per action': FRAME_PER_ACTION,
                'ACCELERATE': ACCELERATE,
                'Penalty': PENALTY
                # TODO: Save learning rate scheduler state_dict()

            }
            if not os.path.isdir('/content/drive/MyDrive/Deep_Learning/DeepDinoAri/DeepDinoRun/checkpoints'):
                os.mkdir('/content/drive/MyDrive/Deep_Learning/DeepDinoAri/DeepDinoRun/checkpoints')
            torch.save(state,
                       f'/content/drive/MyDrive/Deep_Learning/DeepDinoAri/DeepDinoRun/checkpoints/{model_name}_{game_state.games_num}_games_{date_time}.pth')

        t_total += 1
        time.sleep(DELAY_TIME)

        # print out info
        # print("Time = ", t_total)

        if t > OBSERVE and t_total > INITIAL_OBSERVE:  # if t > OBSERVE, start training
            # game_state.pause_game()
            model.train()
            for episode in range(TRAIN):
                print(f"---- Entering Training Episode {episode} ----")
                # sample a random training batch
                # TODO: Add Prioritized Replay
                train_batch = random.sample(D, BATCH)
                # inputs = torch.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32 x 4 x 80 x 80
                targets = torch.zeros(BATCH)  # 32 x 1
                predicted = torch.zeros(BATCH)  # 32 x 1

                # train on the batch: for each experience, extract the experience tuple, create input (state) and targets
                # (predicted q values with reward

                # tic_forward = time.time()

                for i in range(0, BATCH):
                    state_t = train_batch[i][0]
                    action_t = train_batch[i][1]
                    reward_t = train_batch[i][2]
                    state_next = train_batch[i][3]
                    terminal = train_batch[i][4]

                    # inputs[i:i + 1] = state_t.to(device, dtype=torch.float32)  # TODO: make sure this really loads to the device

                    model_predictions = model(
                        state_t.to(device, dtype=torch.float32))  # predicted q values for all actions

                    predicted[i] = model_predictions[0, action_t]  # prediction for the chosen action
                    # TODO: Detach the values for the Q_next_state?? no grad on this value?

                    Q_next_state = model(
                        state_next.to(device, dtype=torch.float32)).detach()  # predicted q values for next state

                    if terminal:
                        targets[i] = reward_t
                    else:
                        targets[i] = reward_t + GAMMA * torch.max(Q_next_state)

                # toc_forward = time.time()
                # print(f'forward time: {toc_forward - tic_forward}')
                targets = targets.to(device, dtype=torch.float32)
                predicted = predicted.to(device, dtype=torch.float32)

                # train using targets and inputs, get loss
                tic_loss = time.time()
                loss = criterion(predicted,
                                 targets)  # TODO: Train only on the chosen action, that we know the reward for?
                toc_loss = time.time()
                optimizer.zero_grad()
                toc_zero_grad = time.time()
                loss.backward()
                toc_backward = time.time()
                optimizer.step()
                toc_step = time.time()
                # print(f"loss time: {toc_loss - tic_loss}, zero grad time: {toc_zero_grad - toc_loss}, loss.backward time: { toc_backward - toc_zero_grad}, steip time: {toc_step - toc_backward}")

                running_loss += loss.data.item()  # TODO: Add some normalization?

                # reset observation counter
                # if t > OBSERVE + TRAIN:
                #     t = 0
                # print("---- Finished Training Episode ----")

            # reset time counter
            t = 0
            # resume the game
            # game_state.resume_game()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
model, image_size = initialize_model(MODEL_NAME, ACTIONS, FEATURE_EXTRACT, use_pretrained)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# TODO: Add learning rate scheduler
game = Game(image_size)
print("Game object created")
dino = DinoAgent(game)
print("Dino Agent Created")
game_state = GameState(dino, game)
print("Game State Created")

last_time = time.time()

# load checkpoint
if CHECKPOINT:  # TODO: make a more generic loading (check if file exists etc.)
    checkpoint = torch.load(SAVE_PATH + '/' + CHECKPOINT_NAME)
    model.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epsilon = checkpoint['epsilon']
    D = checkpoint['experience']  # experience memory
    t_total = checkpoint['t_total']
    game_state.games_num = checkpoint['games num']
    scores = checkpoint['scores']
    iterations = checkpoint['iterations']

    # TODO: load schedulers
else:
    D = deque()  # TODO: Change Replay Memory to enable working with dataloaders?
    epsilon = INITIAL_EPSILON
    t_total = 0
    scores = []
    iterations = []

# TODO: Add data augmentations
# initialise: perform first action and get initial state
do_nothing = torch.zeros(ACTIONS)
do_nothing[0] = 1

# get initial state
x_t, r_0, terminal, score = game_state.get_state(do_nothing)

# stack the first image 4 times as placeholder
s_t = torch.stack((x_t, x_t, x_t, x_t), dim=2)  # 80x80x4
s_t = s_t.permute(2, 0, 1)  # change the axis such that the channels is the first dimension: 4x80x80
# reshape to add a fourth dimension
s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1x4x80x80  // TODOL change to unsqueeze(0)
initial_state = s_t

if ONLY_OBSERVE:  # Load current weights and observe
    OBSERVE = 999999999
    epsilon = FINAL_EPSILON

else:
    OBSERVE = OBSERVATION

t = 0
tic_framerate = time.time()
while True:
    model.eval()
    running_loss = 0.0
    Q_next_state = 0
    chosen_action = 0  # TODO: Change this to enable same action over 4 frames?
    r_t = 0
    a_t = torch.zeros(ACTIONS)  # TODO: Change this to enable same action over 4 frames?
    # chose an epsilon greedy action
    if t % FRAME_PER_ACTION == 0:
        if random.random() <= epsilon:
            # print("-----Random Action-----")
            chosen_action = random.randrange(ACTIONS)
        else:
            q = model(s_t.to(device, dtype=torch.float32))
            chosen_action = torch.argmax(q)
        a_t[chosen_action] = 1

    # reduce epsilon
    if epsilon > FINAL_EPSILON and t_total > OBSERVE:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

    # run the chosen action and observe next state and reward
    x_next, r_t, terminal, score = game_state.get_state(a_t)
    x_next = x_next.reshape(1, 1, x_next.shape[0], x_next.shape[1])  # 1x1x80x80
    s_next = torch.cat((x_next, s_t[:, :3, :, :]), dim=1)  # 1x4x80x80

    # save experience to replay memory, check memory length
    D.append((s_t, chosen_action, r_t, s_next, terminal))
    if len(D) > REPLAY_MEMORY:
        D.popleft()

    # update the state s_t, if the state was terminal, restart the game
    s_t = initial_state if terminal else s_next
    if terminal:
        toc_framerate = time.time()
        last_time = iterations[-1] if len(iterations) > 0 else 0
        framerate = (t_total - last_time) / (toc_framerate - tic_framerate)
        scores.append(score)  # save current game score
        iterations.append(t_total)  # save current number of iterations
        print(
            f' |||| Game: {game_state.games_num} || Score: {score} || Max Score: {np.max(scores)} || Mean of last 10 games: {np.mean(scores[np.max([0, game_state.games_num - 10]):]):.2f} || Epsilon: {epsilon:.2f} || Time: {t_total} || Avarage Framerate: {framerate:.2f} ||||')
        # print(f'Max Score: {np.max(scores)}')
        # print(f'Mean of last 10 games: {np.mean(scores[np.max([0, game_state.games_num-10]):]):.2f}')
        # print(f'Epsilon: {epsilon:.2f}')
        # print("Time = ", t_total)
        # print(f'Avarage Framerate: {framerate:.2f}')
        # time.sleep(3)
        tic_framerate = time.time()

    # update t
    t += 1
    if terminal and game_state.games_num % CHECKPOINT_TIME == 0:
        now = datetime.now()
        date_time = now.strftime("%d_%m_%Y_%H-%M")
        print("Saving Model...")
        state = {
            'net': model.state_dict(),
            'games num': game_state.games_num,
            'optimizer': optimizer.state_dict(),
            't_total': t_total,
            'scores': scores,
            'iterations': iterations,
            'experience': D,
            'epsilon': epsilon,
            'learning rate': LEARNING_RATE,
            'batch size': BATCH,
            'initial epsilon': INITIAL_EPSILON,
            'final epsilon': FINAL_EPSILON,
            'EXPLORE': EXPLORE,
            'TRAIN': TRAIN,
            'OBSERVATION': OBSERVATION,
            'frames per action': FRAME_PER_ACTION,
            'ACCELERATE': ACCELERATE,
            'Penalty': PENALTY
            # TODO: Save learning rate scheduler state_dict()

        }
    # if not os.path.isdir(SAVE_PATH):
    #     os.mkdir(SAVE_PATH)
    # torch.save(state, SAVE_PATH + f'/{model_name}_{game_state.games_num}_games_{date_time}.pth')

    t_total += 1
    time.sleep(DELAY_TIME)

    # print out info
    # print("Time = ", t_total)

    if t > OBSERVE and t_total > INITIAL_OBSERVE:  # if t > OBSERVE, start training
        game_state.pause_game()
        model.train()
        for episode in range(TRAIN):
            print(f"---- Entering Training Episode {episode} ----")
            # sample a random training batch
            # TODO: Add Prioritized Replay
            train_batch = random.sample(D, BATCH)
            # inputs = torch.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32 x 4 x 80 x 80
            targets = torch.zeros(BATCH)  # 32 x 1
            predicted = torch.zeros(BATCH)  # 32 x 1

            # train on the batch: for each experience, extract the experience tuple, create input (state) and targets
            # (predicted q values with reward

            # tic_forward = time.time()

            for i in range(0, BATCH):
                state_t = train_batch[i][0]
                action_t = train_batch[i][1]
                reward_t = train_batch[i][2]
                state_next = train_batch[i][3]
                terminal = train_batch[i][4]

                # inputs[i:i + 1] = state_t.to(device, dtype=torch.float32)  # TODO: make sure this really loads to the device

                model_predictions = model(state_t.to(device, dtype=torch.float32))  # predicted q values for all actions

                predicted[i] = model_predictions[0, action_t]  # prediction for the chosen action
                # TODO: Detach the values for the Q_next_state?? no grad on this value?

                Q_next_state = model(
                    state_next.to(device, dtype=torch.float32)).detach()  # predicted q values for next state

                if terminal:
                    targets[i] = reward_t
                else:
                    targets[i] = reward_t + GAMMA * torch.max(Q_next_state)

            # toc_forward = time.time()
            # print(f'forward time: {toc_forward - tic_forward}')
            targets = targets.to(device, dtype=torch.float32)
            predicted = predicted.to(device, dtype=torch.float32)

            # train using targets and inputs, get loss
            tic_loss = time.time()
            loss = criterion(predicted, targets)  # TODO: Train only on the chosen action, that we know the reward for?
            toc_loss = time.time()
            optimizer.zero_grad()
            toc_zero_grad = time.time()
            loss.backward()
            toc_backward = time.time()
            optimizer.step()
            toc_step = time.time()
            # print(f"loss time: {toc_loss - tic_loss}, zero grad time: {toc_zero_grad - toc_loss}, loss.backward time: { toc_backward - toc_zero_grad}, steip time: {toc_step - toc_backward}")

            running_loss += loss.data.item()  # TODO: Add some normalization?

            # reset observation counter
            # if t > OBSERVE + TRAIN:
            #     t = 0
            # print("---- Finished Training Episode ----")

        # reset time counter
        t = 0
        # resume the game
        game_state.resume_game()
