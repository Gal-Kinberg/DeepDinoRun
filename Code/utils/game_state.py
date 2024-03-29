import torch
from Code.config import *

class GameState:

    def __init__(self, agent, game):
        self._game = game
        self._agent = agent
        self.games_num = 0

    def get_state(self, actions):
        score = self._game.get_score()
        image = self._game.get_screen()
        if not (self._agent.is_dead()):
            reward = 0.1  # survival reward
            is_over = False
            if actions[1] == 1:
                self._agent.jump()
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
