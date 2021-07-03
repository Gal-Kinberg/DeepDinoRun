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
from config import *
from Code.utils.dqn_model import DQN, DuelingDQN

### TRAINING PARAMETERS
# TODO: Move all parameters and hyper-parameters to a config file
# PARAMETERS
CHECKPOINT = False
ACTIONS = 2  # nop, jump, duck
ONLY_OBSERVE = False
OBSERVATION = 600  # timesteps to observe before training
TRAIN = 300
EXPLORE = 100_000  # time boundary for modifying epsilon
INITIAL_OBSERVE = 5000
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.001  # TODO: use a PyTorch scheduler for epsilon? Cosine?
REPLAY_MEMORY = 50000  # number of memories to keep
img_channels = 4  # stacking 4 images together

# Training Hyper-parameters
ACCELERATE = False
PENALTY = False
use_pretrained = True
CHECKPOINT_TIME = 100
DELAY_TIME = 0.02
MODEL_NAME = "dueling dqn"
FEATURE_EXTRACT = True
BATCH = 64  # training batch size
FRAME_PER_ACTION = 1  # TODO: Change to 4 frames per action?
LEARNING_RATE = 4e-5
GAMMA = 0.99  # decay rate of past observations

RUN_NAME = "No Acceleration, Normalized"

# class Game:
#     def __init__(self):
#         self.driver = webdriver.Chrome(PATH)
#         self.driver.set_window_position(x=-10, y=0)
#         self.driver.get(url1)
#         print('Game Loaded')
#         time.sleep(0.5)
#         self.driver.execute_script(init_script2)
#         self.game = self.driver.find_element_by_id("t")
#
#     def restart(self):
#         self.driver.execute_script("Runner.instance_.restart()")
#
#     def press_up(self):
#         self.game.send_keys(Keys.ARROW_UP)
#
#     def press_down(self):
#         self.game.send_keys(Keys.ARROW_DOWN)
#
#     def get_score(self):
#         score_array = self.driver.execute_script("return Runner.instance_.distanceMeter.digits")
#         score = ''.join(score_array)
#         return int(score)
#
#     # TODO: Make screen grabbing better
#     def get_screen(self):
#         image_b64 = self.driver.execute_script(getScreenScript)
#         screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
#         # print('Captured Screen')
#         # plt.imshow(screen)
#         # plt.show()
#         return process_img(screen)
#
#     def is_crashed(self):
#         return self.driver.execute_script("return Runner.instance_.crashed")
#
#     def is_playing(self):
#         return self.driver.execute_script("return Runner.instance_.playing")
#
#     def pause(self):
#         return self.driver.execute_script("return Runner.instance_.stop()")
#
#     def resume(self):
#         return self.driver.execute_script("return Runner.instance_.play()")
#
#     def end(self):
#         self.driver.close()

### Game Class
class Game:
    def __init__(self, image_size):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(PATH, chrome_options=chrome_options)
        self.driver.set_window_position(x=-10, y=0)
        self.driver.get(url1)
        print('Game Loaded')
        time.sleep(0.5)
        if not ACCELERATE:
          self.driver.execute_script("Runner.config.ACCELERATION=0")
        self.driver.execute_script("Runner.config.CLEAR_TIME = 0")
        self.driver.execute_script(init_script2)
        self.game = self.driver.find_element_by_id("t")
        self.image_size = image_size

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
        return process_img(screen, self.image_size)

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

# class DinoAgent:
#     def __init__(self, game):
#         self._game = game
#         self.jump()  # jump once to start the game
#
#     def jump(self):
#         self._game.press_up()
#
#     def duck(self):
#         self._game.press_down()
#
#     def is_running(self):
#         return self._game.is_playing()
#
#     def is_dead(self):
#         return self._game.is_crashed()

### Dino Agent Class
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


# class GameState:
#     def __init__(self, agent, game):
#         self._game = game
#         self._agent = agent
#         # -- create display for images
#
#     def get_state(self, actions):
#         score = self._game.get_score()
#         image = self._game.get_screen()
#         global games_num
#         # -- display the image
#         if not (self._agent.is_dead()):
#             reward = 0.1  # survival reward
#             is_over = False
#             if actions[1] == 1:
#                 self._agent.jump()
#                 # TODO: Remove jumping penalty?
#                 # reward = -0.1  # jumping is expensive
#             elif ACTIONS == 3 and actions[2] == 1:
#                 self._agent.duck()
#         else:
#             # -- save the score
#             games_num += 1
#             print("Game Number: ", games_num)
#             reward = -1  # punishment for dying
#             is_over = True
#             self._game.restart()
#         return torch.from_numpy(image), reward, is_over  # convert picture to Tensor

### Game-State Class
class GameState:

    def __init__(self, agent, game):
        self._game = game
        self._agent = agent
        self.games_num = 0
        # self.display = show_img() #display the processed image on screen using openCV, implemented using python coroutine
        # self.display.__next__() # initiliaze the display coroutine
        # -- create display for images

    def get_state(self, actions):
        score = self._game.get_score()
        image = self._game.get_screen()
        # -- display the image
        # self.display.send(image) #display the image on screen
        if not (self._agent.is_dead()):
            reward = 0.1  # survival reward
            is_over = False
            if actions[1] == 1:
                self._agent.jump()
                # TODO: Remove jumping penalty?
                if PENALTY:
                    reward = 0.0  # jumping is expensive
            elif ACTIONS == 3 and actions[2] == 1:
                self._agent.duck()
        else:
            # -- save the score
            self.games_num += 1
            # print("Game Number: ", self.games_num)
            reward = -1  # punishment for dying
            is_over = True
            self._game.restart()
        return torch.from_numpy(image), reward, is_over, score  # convert picture to Tensor

    def pause_game(self):
        self._game.pause()

    def resume_game(self):
        self._game.resume()

# # TODO: better processing
# def process_img(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to GreyScale
#     image = image[:300, :500]
#     image = cv2.resize(image, (80, 80))
#     # plt.imshow(image, cmap = 'gray')
#     # plt.show()
#     return image

### Screen Grabbing and Processing Function
def process_img(image, image_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to GreyScale
    image = image[:300, :500]
    image = cv2.resize(image, (image_size, image_size))
    # plt.imshow(image, cmap = 'gray')
    # plt.show()
    return image


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

class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class DQN_Conv_Block(nn.Module):
  def __init__(self):
    super(DQN_Conv_Block, self).__init__()
    self.conv_layer = nn.Sequential(

            # TODO: Change to kernel sizes of 3 and no strides?
            # TODO: Fix dimensions of convolution layers
            # Conv Block (Feature Extraction)
            nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

  def forward(self, x):
      return self.conv_layer(x)
# class DQN(nn.Module):
#     """DQN Network for the Dino Run Game"""
#
#     def __init__(self, ACTIONS):
#         super(DQN, self).__init__()
#         self.conv_layer = nn.Sequential(
#
#             # TODO: Change to kernel sizes of 3 and no strides?
#             # TODO: Fix dimensions of convolution layers
#             # Conv Block (Feature Extraction)
#             nn.Conv2d(in_channels=img_channels, out_channels=32, kernel_size=8, stride=4),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.ReLU(inplace=True)
#         )
#
#         self.fc_layer = nn.Sequential(
#             nn.Linear(3096, 512),  # TODO: Fill in true value of flattened layer
#             nn.ReLU(inplace=True),
#             nn.Linear(512, ACTIONS)
#         )
#
#     def forward(self, x):
#         # conv layer
#         x = self.conv_layer(x)
#
#         # flatten
#         x = x.view(x.size(0), -1)
#
#         # fully connected
#         x = self.fc_layer(x)
#
#         return x
### Models
# class DQN(nn.Module):
#     """DQN Network for the Dino Run Game"""
#
#     def __init__(self, ACTIONS):
#         super(DQN, self).__init__()
#         self.conv_layer = DQN_Conv_Block()
#
#         self.fc_layer = nn.Sequential(
#             nn.Linear(4096, 512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, ACTIONS)
#         )
#
#     def forward(self, x):
#         x = x.float() / 255
#
#         # conv layer
#         features = self.conv_layer(x)
#
#         # flatten
#         features = features.view(features.size(0), -1)
#         # print(f'features size: {features.size()}')
#
#         # fully connected
#         q_vals = self.fc_layer(features)
#
#         return q_vals

# class DuelingDQN(nn.Module):
#     """Basic Dueling DQN Network"""
#     def __init__(self, ACTIONS):
#         super(DuelingDQN, self).__init__()
#
#         # Conv Block (Feature Extraction)
#         self.conv_layer = DQN_Conv_Block()
#
#         self.value_stream = nn.Sequential(
#             nn.Linear(in_features=4096, out_features=512),
#             nn.ReLU(),
#             nn.Linear(in_features=512, out_features=1)
#         )
#
#         self.advantage_stream = nn.Sequential(
#             nn.Linear(in_features=4096, out_features=512),
#             nn.ReLU(),
#             nn.Linear(in_features=512, out_features=ACTIONS)
#         )
#
#     def forward(self, x):
#         x = x.float() / 255
#         features = self.conv_layer(x)
#
#         # Flatten
#         features = features.view(features.size(0), -1)
#
#         # value and advantage
#         values = self.value_stream(features)
#         advantages = self.advantage_stream(features)
#
#         # estimate Q values
#         q_vals = values + (advantages - advantages.mean())
#
#         return q_vals

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

    elif mode_name == "dueling resnet":
        model_ft = DuelingResent(num_classes)
        input_size = 224
        set_parameter_requires_grad(model_ft.conv_layer, feature_extracting=True)


    else:
        raise NotImplementedError

    return model_ft, input_size



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


# def train_network(model, game_state, device, criterion, optimizer, observe=False):
#     last_time = time.time()
#     # -- load previous replay memory if exists
#     D = deque()  # TODO: Change Replay Memory to enable working with dataloaders?
#     # TODO: Add data augmentations
#     # initialise: perform first action and get initial state
#     do_nothing = torch.zeros(ACTIONS)
#     do_nothing[0] = 1
#
#     # get initial state
#     x_t, r_0, terminal = game_state.get_state(do_nothing)
#
#     # stack the first image 4 times as placeholder
#     s_t = torch.stack((x_t, x_t, x_t, x_t), dim=2)
#     # reshape to add a fourth dimension
#     s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])
#     initial_state = s_t
#
#     # TODO: Add model initialization from checkpoint, if exists
#     if observe:  # Load current weights and observe
#         OBSERVE = 999999999
#         epsilon = FINAL_EPSILON
#         # print("Loading weights")
#         # model.load_weights("model.h5")
#         # adam = Adam(lr=LEARNING_RATE)
#         # model.compile(loss='mse', optimizer=adam)
#         # print("Weights loaded successfully")
#     else:
#         OBSERVE = OBSERVATION
#         # epsilon = load_obj("epsilon") # ??
#         epsilon = INITIAL_EPSILON
#         # model.load_weights("model.h5")
#         # adam = Adam(lr=LEARNING_RATE)
#         # model.compile(loss='mse', optimizer=adam)
#
#     # -- load previous time
#     # t = load_obj("time")
#     t = 0
#
#     while True:  # TODO: Change to a concrete number of maximal time steps
#         model.train()
#         running_loss = 0.0
#         Q_next_state = 0
#         chosen_action = 0  # TODO: Change this to enable same action over 4 frames?
#         r_t = 0
#         a_t = torch.zeros(ACTIONS)  # TODO: Change this to enable same action over 4 frames?
#         # chose an epsilon greedy action
#         if t % FRAME_PER_ACTION == 0:
#             if random.random() <= epsilon:
#                 print("-----Random Action-----")
#                 chosen_action = random.randrange(ACTIONS)
#             else:
#                 q = model(s_t.to(device))
#                 chosen_action = torch.argmax(q)
#             a_t[chosen_action] = 1
#
#         # reduce epsilon
#         if epsilon > FINAL_EPSILON and t > OBSERVE:
#             epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
#
#         # run the chosen action and observe next state and reward
#         x_next, r_t, terminal = game_state.get_state(a_t)
#         x_next = x_next.reshape(1, x_next.shape[0], x_next.shape[1], 1)
#         s_next = torch.cat((x_next, s_t[:, :, :, :3]), dim=3) # TODO: Change to torch.stack like before?
#
#         # save experience to replay memory, check memory length
#         D.append((s_t, chosen_action, r_t, s_next, terminal))
#         if len(D) > REPLAY_MEMORY:
#             D.popleft()
#
#         if t > OBSERVE:  # if t > OBSERVE, start training
#             # print("---- Entering Training Episode ----")
#             # sample a random training batch
#             # TODO: Add Prioritized Replay
#             train_batch = random.sample(D, BATCH)
#             inputs = torch.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32 x 80 x 80 x 4
#             targets = torch.zeros((BATCH, ACTIONS))  # 32 x 2
#             predicted = torch.zeros((BATCH, ACTIONS))
#
#             # train on the batch: for each experience, extract the experience tuple, create input (state) and targets
#             # (predicted q values with reward
#             for i in range(0, BATCH):
#                 state_t = train_batch[i][0]
#                 action_t = train_batch[i][1]
#                 reward_t = train_batch[i][2]
#                 state_next = train_batch[i][3]
#                 terminal = train_batch[i][4]
#
#                 inputs[i:i + 1] = state_t.to(device)  # TODO: make sure this really loads to the device
#                 predicted[i] = model(state_t.to(device))  # predicted q values for all actions
#                 Q_next_state = model(state_next.to(device))  # predicted q values for next state
#
#                 if terminal:
#                     targets[i, action_t] = reward_t
#                 else:
#                     targets[i, action_t] = reward_t + GAMMA * torch.max(Q_next_state)
#
#             targets = targets.to(device)
#
#             # train using targets and inputs, get loss
#             loss = criterion(predicted, targets)  # TODO: Train only on the chosen action, that we know the reward for?
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.data.item()  # TODO: Add some normalization?
#             # loss += model.train_on_batch(inputs, targets)
#
#             # reset observation counter
#             # if t > OBSERVE + TRAIN:
#             #     t = 0
#             # print("---- Finished Training Episode ----")
#
#         # update the state s_t, if the state was terminal, restart the game
#         s_t = initial_state if terminal else s_next
#         # TODO: Keep the score of the game
#         # update t
#         t += 1
#         if t % 5000 == 0:
#             model.save_weights('model_save.h5')
#
#         # print out info
#         # print("Time = ", t)


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

# def playGame(observe=False):
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     criterion = nn.MSELoss()
#     model = DQN(ACTIONS).to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#     # TODO: Add learning rate scheduler
#     game = Game()
#     print("Game object created")
#     dino = DinoAgent(game)
#     print("Dino Agent Created")
#     game_state = GameState(dino, game)
#     print("Game State Created")
#     try:
#         train_network(model, game_state, device, criterion, optimizer, observe=False)
#     except StopIteration:
#         game.end()


### Run Game Function
def playGame(observe=False, use_pretrained=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.MSELoss()
    model, image_size = initialize_model(MODEL_NAME, ACTIONS, FEATURE_EXTRACT, use_pretrained)
    model = model.to(device)
    # for param in model.fc_layer.parameters():
    #   print(param.requires_grad)
    # print(f'model: {model}')
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # TODO: Add learning rate scheduler
    game = Game(image_size)
    print("Game object created")
    dino = DinoAgent(game)
    print("Dino Agent Created")
    game_state = GameState(dino, game)
    print("Game State Created")
    try:
        train_network(model, MODEL_NAME + " No Acceleration, Split Observe-Train", game_state, device, criterion, optimizer, observe)
    except StopIteration:
        game.end()

# # TODO: Move all parameters and hyper-parameters to a config file
# # PARAMETERS
# ACTIONS = 2  # nop, jump, duck
# OBSERVATION = 300.0  # timesteps to observe before training
# TRAIN = 300.0
# EXPLORE = 100000  # time boundary for modifying epsilon
# INITIAL_EPSILON = 0.6
# FINAL_EPSILON = 0.0001
# REPLAY_MEMORY = 50000  # number of memories to keep
# img_rows, img_cols = 80, 80
# img_channels = 4  # stacking 4 images together
#
# # Training Hyper-parameters
# BATCH = 16  # training batch size
# FRAME_PER_ACTION = 1  # TODO: Change to 4 frames per action?
# LEARNING_RATE = 1e-4
# GAMMA = 0.99  # decay rate of past observations

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
model, image_size = initialize_model(MODEL_NAME, ACTIONS, FEATURE_EXTRACT, use_pretrained)
model = model.to(device)
# for param in model.fc_layer.parameters():
#   print(param.requires_grad)
# print(f'model: {model}')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# TODO: Add learning rate scheduler
game = Game(image_size)
print("Game object created")
dino = DinoAgent(game)
print("Dino Agent Created")
game_state = GameState(dino, game)
print("Game State Created")

# model_name = MODEL_NAME + " " + RUN_NAME
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


# if __name__ == "__main__":
#     playGame(observe=False)

