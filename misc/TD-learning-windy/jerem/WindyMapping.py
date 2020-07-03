import numpy as np
import WindyWorld


class WindyProblem:
    """
    Maps a reinforcement learning algorithm to WindyWorld.Application.
    """
    def __init__(self, algo, size_x, size_y, init_pos, end, wind, delay, show=True, nb_win=100):
        self.x = size_x
        self.y = size_y
        self.delay = delay
        self.algo = algo

        self.terminal_state = end

        self.state = init_pos
        self.prev_state = self.state
        self.istate = self.state

        self.app = WindyWorld.Application(size_x, size_y, init_pos, end, wind)

        self.steps = np.zeros(nb_win)

        self.show = show
        self.nb_win = nb_win
        self.count_win = 0

        if self.show:
            self.app.root.after(200, self.play)
            self.app.start()
        else:
            while self.count_win < self.nb_win:
                self.play()
            self.app.root.destroy()

    def play(self):
        action = self.algo.choose_action(self.state)
        self.app.move(action[0], action[1])

        self.prev_state = self.state
        self.state = (self.app.pos[0], self.app.pos[1])

        reward = 0 if self.state == self.terminal_state else -1

        self.algo.fit_step(self.prev_state, action, reward, self.state)

        if self.state == self.terminal_state:
            self.steps[self.count_win] = self.app.steps
            self.count_win += 1
            self.app.reset()

        if self.show and self.count_win < self.nb_win:
            self.app.root.after(self.delay, self.play)
