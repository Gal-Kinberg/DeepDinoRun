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
