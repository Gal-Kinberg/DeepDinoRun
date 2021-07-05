from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import time
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2
from Code.config import *


class Game:
    def __init__(self, image_size):
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        self.driver = webdriver.Chrome(PATH, chrome_options=chrome_options)
        print("Loading Game")
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


def process_img(image, image_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert to GreyScale
    image = image[:300, :500]
    image = cv2.resize(image, (image_size, image_size))
    # plt.imshow(image, cmap = 'gray')
    # plt.show()
    return image
