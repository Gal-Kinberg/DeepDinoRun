# Paths and URLs
PATH = "C:\\Users\\arifr\\OneDrive\\Desktop\\chromedriver.exe"
url1 = "https://chromedino.com"
url2 = "chrome://dino/"
loss_file_path = "./model/loss_df.csv"

# Java scripts
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'; Cloud.config.WIDTH = 0"
init_script2 = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas';"
getScreenScript = "canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL(" \
                  ").substring(22)"

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
CHECKPOINT_TIME = 10
DELAY_TIME = 0.02
MODEL_NAME = "dueling dqn"
FEATURE_EXTRACT = True
BATCH = 64  # training batch size
FRAME_PER_ACTION = 1  # TODO: Change to 4 frames per action?
LEARNING_RATE = 4e-5
GAMMA = 0.99  # decay rate of past observations
RUN_NAME = "No Acceleration, Normalized"
