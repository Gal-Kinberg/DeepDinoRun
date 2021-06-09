# Selenium Imports
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

# PyTorch Imports
import torch
from torch import nn

# Python Imports
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import deque

# Extra Imports
from io import BytesIO
import base64
import pickle
import json
import cv2
from PIL import Image

# Import config for paths and scripts
from config import *


class Game:
    def __init__(self):
        self.driver = webdriver.Chrome(PATH)
        self.driver.set_window_position(x=-10, y=0)
        self.driver.get(url1)
        print('Game Loaded')
        time.sleep(0.5)
        self.driver.execute_script(init_script2)
        self.game = self.driver.find_element_by_id("t")

    def restart(self):
        self.driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        self.game.send_keys(Keys.ARROW_UP)

    def press_down(self):
        self.game.send_keys(Keys.ARROW_DOWN)

    def get_score(self):
        score_array = self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        return int(score)

    # TODO: Make screen grabbing better
    def get_screen(self):
        image_b64 = self.driver.execute_script(getScreenScript)
        screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        # print('Captured Screen')
        # plt.imshow(screen)
        # plt.show()
        return process_img(screen)

    def is_crashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed")

    def is_playing(self):
        return self.driver.execute_script("return Runner.instance_.playing")

    def pause(self):
        return self.driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        return self.driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self.driver.close()


class DinoAgent:
    def __init__(self, game):
        self._game = game
        self.jump()  # jump once to start the game

    def jump(self):
        self._game.press_up()

    def duck(self):
        self._game.press_down()

    def is_running(self):
        return self._game.is_playing()

    def is_dead(self):
        return self._game.is_crashed()


# TODO: Change this to a class property
games_num = 0


class GameState:
    def __init__(self, agent, game):
        self._game = game
        self._agent = agent
        # -- create display for images

    def get_state(self, actions):
        score = self._game.get_score()
        image = self._game.get_screen()
        global games_num
        # -- display the image
        if not (self._agent.is_dead()):
            reward = 0.1  # survival reward
            is_over = False
            if actions[1] == 1:
                self._agent.jump()
                # TODO: Remove jumping penalty?
                # reward = -0.1  # jumping is expensive
            elif ACTIONS == 3 and actions[2] == 1:
                self._agent.duck()
        else:
            # -- save the score
            games_num += 1
            print("Game Number: ", games_num)
            reward = -1  # punishment for dying
            is_over = True
            self._game.restart()
        return torch.from_numpy(image), reward, is_over  # convert picture to Tensor


# TODO: better processing
def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to GreyScale
    image = image[:300, :500]
    image = cv2.resize(image, (80, 80))
    # plt.imshow(image, cmap = 'gray')
    # plt.show()
    return image


class DQN(nn.Module):
    """DQN Network for the Dino Run Game"""

    def __init__(self, ACTIONS):
        super(DQN, self).__init__()
        self.conv_layer = nn.Sequential(

            # TODO: Change to kernel sizes of 3 and no strides?
            # TODO: Fix dimensions of convolution layers
            # Conv Block (Feature Extraction)
            nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(3096, 512),  # TODO: Fill in true value of flattened layer
            nn.ReLU(inplace=True),
            nn.Linear(512, ACTIONS)
        )

    def forward(self, x):
        # conv layer
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fully connected
        x = self.fc_layer(x)

        return x


# Keras Version:
# def build_model():
#     print("Building model")
#     model = Sequential()
#     model.add(
#         Conv2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=(img_cols, img_rows, img_channels)))  # 80*80*4
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (4, 4), strides=(2, 2), padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Activation('relu'))
#     model.add(Flatten())
#     model.add(Dense(512))
#     model.add(Activation('relu'))
#     model.add(Dense(ACTIONS))
#     adam = Adam(lr=LEARNING_RATE)
#     model.compile(loss='mse', optimizer=adam)
#
#     # create model file if not present
#     if not os.path.isfile(loss_file_path):
#         model.save_weights('model.h5')
#     print("Model built successfully!")
#     return model


def train_network(model, game_state, device, criterion, optimizer, observe=False):
    last_time = time.time()
    # -- load previous replay memory if exists
    D = deque()  # TODO: Change Replay Memory to enable working with dataloaders?
    # TODO: Add data augmentations
    # initialise: perform first action and get initial state
    do_nothing = torch.zeros(ACTIONS)
    do_nothing[0] = 1

    # get initial state
    x_t, r_0, terminal = game_state.get_state(do_nothing)

    # stack the first image 4 times as placeholder
    s_t = torch.stack((x_t, x_t, x_t, x_t), dim=2)
    # reshape to add a fourth dimension
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
    initial_state = s_t

    # TODO: Add model initialization from checkpoint, if exists
    if observe:  # Load current weights and observe
        OBSERVE = 999999999
        epsilon = FINAL_EPSILON
        # print("Loading weights")
        # model.load_weights("model.h5")
        # adam = Adam(lr=LEARNING_RATE)
        # model.compile(loss='mse', optimizer=adam)
        # print("Weights loaded successfully")
    else:
        OBSERVE = OBSERVATION
        # epsilon = load_obj("epsilon") # ??
        epsilon = INITIAL_EPSILON
        # model.load_weights("model.h5")
        # adam = Adam(lr=LEARNING_RATE)
        # model.compile(loss='mse', optimizer=adam)

    # -- load previous time
    # t = load_obj("time")
    t = 0

    while True:  # TODO: Change to a concrete number of maximal time steps
        model.train()
        running_loss = 0.0
        Q_next_state = 0
        chosen_action = 0  # TODO: Change this to enable same action over 4 frames?
        r_t = 0
        a_t = torch.zeros(ACTIONS)  # TODO: Change this to enable same action over 4 frames?
        # chose an epsilon greedy action
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("-----Random Action-----")
                chosen_action = random.randrange(ACTIONS)
            else:
                q = model(s_t.to(device))
                chosen_action = torch.argmax(q)
            a_t[chosen_action] = 1

        # reduce epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the chosen action and observe next state and reward
        x_next, r_t, terminal = game_state.get_state(a_t)
        x_next = x_next.reshape(1, x_next.shape[0], x_next.shape[1], 1)
        s_next = torch.cat((x_next, s_t[:, :, :, :3]), dim=3) # TODO: Change to torch.stack like before?

        # save experience to replay memory, check memory length
        D.append((s_t, chosen_action, r_t, s_next, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        if t > OBSERVE:  # if t > OBSERVE, start training
            # print("---- Entering Training Episode ----")
            # sample a random training batch
            # TODO: Add Prioritized Replay
            train_batch = random.sample(D, BATCH)
            inputs = torch.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32 x 80 x 80 x 4
            targets = torch.zeros((BATCH, ACTIONS))  # 32 x 2
            predicted = torch.zeros((BATCH, ACTIONS))

            # train on the batch: for each experience, extract the experience tuple, create input (state) and targets
            # (predicted q values with reward
            for i in range(0, BATCH):
                state_t = train_batch[i][0]
                action_t = train_batch[i][1]
                reward_t = train_batch[i][2]
                state_next = train_batch[i][3]
                terminal = train_batch[i][4]

                inputs[i:i + 1] = state_t.to(device)  # TODO: make sure this really loads to the device
                predicted[i] = model(state_t.to(device))  # predicted q values for all actions
                Q_next_state = model(state_next.to(device))  # predicted q values for next state

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * torch.max(Q_next_state)

            targets = targets.to(device)

            # train using targets and inputs, get loss
            loss = criterion(predicted, targets)  # TODO: Train only on the chosen action, that we know the reward for?
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()  # TODO: Add some normalization?
            # loss += model.train_on_batch(inputs, targets)

            # reset observation counter
            # if t > OBSERVE + TRAIN:
            #     t = 0
            # print("---- Finished Training Episode ----")

        # update the state s_t, if the state was terminal, restart the game
        s_t = initial_state if terminal else s_next
        # TODO: Keep the score of the game
        # update t
        t += 1
        if t % 5000 == 0:
            model.save_weights('model_save.h5')

        # print out info
        # print("Time = ", t)


def playGame(observe=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    model = DQN(ACTIONS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # TODO: Add learning rate scheduler
    game = Game()
    print("Game object created")
    dino = DinoAgent(game)
    print("Dino Agent Created")
    game_state = GameState(dino, game)
    print("Game State Created")
    try:
        train_network(model, game_state, device, criterion, optimizer, observe=False)
    except StopIteration:
        game.end()


# TODO: Move all parameters and hyper-parameters to a config file
# PARAMETERS
ACTIONS = 2  # nop, jump, duck
OBSERVATION = 300.0  # timesteps to observe before training
TRAIN = 300.0
EXPLORE = 100000  # time boundary for modifying epsilon
INITIAL_EPSILON = 0.6
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000  # number of memories to keep
img_rows, img_cols = 80, 80
img_channels = 4  # stacking 4 images together

# Training Hyper-parameters
BATCH = 16  # training batch size
FRAME_PER_ACTION = 1  # TODO: Change to 4 frames per action?
LEARNING_RATE = 1e-4
GAMMA = 0.99  # decay rate of past observations

#
# game = Game()
# while 1:
#     time.sleep(0.3)
#     game.get_screen()
#     if game.is_crashed():
#         game.restart()
#         print("restarting")
#     else:
#         game.press_up() if random.choice([0, 1]) else (game.press_down() if random.choice([0, 1]) else None)
#         # print("Playing: ", game.is_playing())
#         # print("Crashed: ", game.is_crashed())

if __name__ == "__main__":
    playGame(observe=False)

