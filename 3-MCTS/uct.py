# This is a very simple implementation of the uct Monte Carlo Tree Search
# algorithm in Python 3 (initially in Python 2.7).
# The function uct(rootstate, itermax, verbose = False) is towards the bottom
# of the code.
# It aims to have the clearest and simplest possible code, and for the sake of
# clarity, the code is orders of magnitude less efficient than it could be
# made, particularly by using a state.GetRandomMove() or
# state.DoRandomRollout() function.
#
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse
# (University of York, UK) September 2012.
#
# Adaptation from Python 2.7 to Python 3, and pep8 compliance:
# Valentin Emiya, Aix-Marseille University, 2019.
#
# Licence is granted to freely use and distribute for any sensible/legal
# purpose so long as this comment remains in any distributed code.
#
# For more information about Monte Carlo Tree Search check out our web site at
# www.mcts.ai

from math import sqrt, log
import random


class Node:
    """
    A node in the game tree. Note wins is always from the viewpoint of
    player_just_moved. Crashes if state not specified.
    """
    def __init__(self, move=None, parent=None, state=None):
        # the move that got us to this node - "None" for the root node
        self.move = move
        self.parentNode = parent  # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_moves()  # future child nodes
        # the only part of the state that the Node needs later
        self.player_just_moved = state.player_just_moved

    def uct_select_child(self):
        """
        Use the UCB1 formula to select a child node. Often a constant UCTK is
        applied so we have
        lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits
        to vary the amount of exploration versus exploitation.
        """
        s = sorted(self.childNodes,
                   key=lambda c:
                   c.wins / c.visits + sqrt(2 * log(self.visits) / c.visits))[-1]
        return s

    def add_child(self, m, s):
        """
        Remove m from untriedMoves and add a new child node for this move.
        Return the added child node
        """
        n = Node(move=m, parent=self, state=s)
        self.untried_moves.remove(m)
        self.childNodes.append(n)
        return n

    def use_heuristic_even(self):
        """
        Set the node to Q=1/2, using the Q_even heuristic.
        """
        self.wins = 1
        self.visits = 2

    def use_heuristic_grandfather(self):
        """
        Set the node wins/visits to the wins/visits of his grandfather.
        """
        father = self.parentNode
        if father is not None:
            grandfather = father.parentNode
            if grandfather is not None:
                self.wins = grandfather.wins
                self.visits = grandfather.wins
                return True

        return False

    def update(self, result):
        """
        update this node - one additional visit and result additional wins.
        result must be from the viewpoint of player_just_moved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" \
               + str(self.visits) + " U:" + str(self.untried_moves) + "]"

    def tree_to_string(self, indent):
        s = _indent_string(indent) + str(self)
        for c in self.childNodes:
            s += c.tree_to_string(indent + 1)
        return s

    def children_to_string(self):
        s = ""
        for c in self.childNodes:
            s += str(c) + "\n"
        return s


def _indent_string(indent):
    s = "\n"
    for i in range(1, indent + 1):
        s += "| "
    return s


def uct_search(root_state, iter_max, heuristic=None, verbose=False):
    """
    Conduct a uct search for itermax iterations starting from rootstate.
    Return the best move from the rootstate.
    Assumes 2 alternating players (player 1 starts), with game results in the
    range [0.0, 1.0].
    """
    valid_heuristics = [None, "even", "grandfather"]
    assert heuristic in valid_heuristics,\
        f"Heuristic should be in {valid_heuristics}"

    rootnode = Node(state=root_state)
    if heuristic is "even":
        rootnode.use_heuristic_even()

    for i in range(iter_max):
        node = rootnode
        state = root_state.clone()

        # Select
        # node is fully expanded and non-terminal
        while node.untried_moves == [] and node.childNodes != []:
            node = node.uct_select_child()
            state.do_move(node.move)

        # Expand
        # if we can expand (i.e. state/node is non-terminal)
        if len(node.untried_moves) > 0:
            m = random.choice(node.untried_moves)
            state.do_move(m)
            node = node.add_child(m, state)  # add child and descend tree
            if heuristic is "even":
                node.use_heuristic_even()
            elif heuristic is "grandfather":
                node.use_heuristic_grandfather()

        # Rollout - this can often be made orders of magnitude quicker using a
        # state.GetRandomMove() function
        while len(state.get_moves()) > 0:  # while state is non-terminal
            state.do_move(random.choice(state.get_moves()))

        # Backpropagate
        # backpropagate from the expanded node and work back to the root node
        while node is not None:
            # state is terminal. update node with result from POV of
            # node.player_just_moved
            node.update(state.get_result(node.player_just_moved))
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if verbose:
        print(rootnode.tree_to_string(0))
    # else:
    #    print(rootnode.children_to_string())
    # return the move that was most visited
    return sorted(rootnode.childNodes, key=lambda c: c.visits)[-1].move


if __name__ == "__main__":
    from games import OthelloState, OXOState, NimState

    """
    Play a sample game between two uct players where each player gets a
    different number of uct iterations (= simulations = tree nodes).
    """
    player_2_iter = 1000
    player_1_iter = 100

    # uncomment to play Othello on a square board of the given size
    state = OthelloState()

    # uncomment to play OXO
    # state = OXOState()

    # uncomment to play Nim with the given number of starting chips
    # state = NimState(15)

    while len(state.get_moves()) > 0:
        print(state)
        if state.player_just_moved == 1:
            # Player 2
            m = uct_search(root_state=state,
                           iter_max=player_2_iter,
                           verbose=True)
        else:
            # Player 1
            m = uct_search(root_state=state,
                           iter_max=player_1_iter,
                           verbose=True)
        print("Best Move: " + str(m) + "\n")
        state.do_move(m)

    # print results for that game
    if state.get_result(state.player_just_moved) == 1.0:
        print("Player " + str(state.player_just_moved) + " wins!")
    elif state.get_result(state.player_just_moved) == 0.0:
        print("Player " + str(3 - state.player_just_moved) + " wins!")
    else:
        print("Nobody wins!")
