# This file contains very simple implementation of some games.
# It aims to be the most generic and simplest possible code.
#
# Current available games are :
# ----- WindyEnv
#
# Written by :
# Hippolyte L. DEBERNARDI, Aix-Marseille University, 2020
#
# Licence is granted to freely use and distribute for any sensible/legal
# purpose so long as this comment remains in any distributed code.

import abc
import numpy as np


class Game(abc.ABC):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_actions(self):
        raise NotImplementedError

    @abc.abstractmethod
    def is_done(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def do_action(self, state, action):
        raise NotImplementedError

    @abc.abstractmethod
    def to_string(self, state):
        raise NotImplementedError


class WindyEnv(Game):
    """

    """
    def __init__(self, grid, winds, winning_cell):
        super().__init__()
        self.grid = grid
        self.winds = winds
        self.winning_cell = winning_cell

        self.actions = [
            (-1, 0),  # UP
            (+1, 0),  # DOWN
            (0, +1),  # RIGHT
            (0, -1),  # LEFT
        ]

    def _grid_limit(self, state):
        """
        Ensure we stay inside the grid.
        """
        x, y = state

        # cannot go under grid
        x = min(x, self.grid.shape[0] - 1)
        # cannot go above grid
        x = max(x, 0)

        # cannot go further on right
        y = min(y, self.grid.shape[1] - 1)
        # cannot go further on left
        y = max(y, 0)

        return x, y

    def get_actions(self):
        return self.actions

    def is_done(self, state):
        return self.winning_cell == state

    def do_action(self, state, action):
        """
        Realize one step, i.e. apply `action` to the `state`.

        Return the reward, the new state and a boolean to know if
        we reached the destination.
        """
        x, y = state
        dx, dy = action

        # apply winds
        x -= self.winds[y]

        # ensure that we stay inside the grid
        state = self._grid_limit((x + dx, y + dy))

        # minimize an infinite sum so we can always return a negative value
        reward = -1

        return reward, state, self.is_done(state)

    def to_string(self, state):
        s = ""

        x, y = self.grid.shape

        for line in range(x):
            for column in range(y):
                # add a column break
                s += "|"

                # add the correct tile marker
                if (line, column) == self.winning_cell:
                    s += " G "
                elif (line, column) == state:
                    s += " X "
                else:
                    s += "   "

                # add a column break for last column
                if column == self.grid.shape[1] - 1:
                    s += "|"

            s += "\n"

        # add the wind powers
        for column in range(self.winds.shape[1]):
            s += f"  {self.winds[0][column]} "

        return s


if __name__ == "__main__":
    # ----- Game configuration -----
    grid_size = (7, 10)

    wind_forces = np.zeros(grid_size).astype(int)
    wind_forces[:, [3, 4, 5, 8]] = 1
    wind_forces[:, [6, 7]] = 2

    destination = (3, 7)

    reward_in_case_of_winning = 0
    reward_otherwise = -1

    # ----- Game instantiation -----
    game = WindyEnv(shape=grid_size,
                    winds=wind_forces,
                    winning_cell=destination)

    # ----- Display some infos -----
    source = (3, 0)
    print(f"\n{game.to_string(source)}")

    actions = game.get_actions()
    print(f"\nActions : {actions}")
