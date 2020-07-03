# Example GameState classes for Nim, OXO and Othello are included to give some
# idea of how you can write your own GameState use uct in your 2-player
# game.
# 
# Written by Peter Cowling, Ed Powley, Daniel Whitehouse
# (University of York, UK) September 2012.
#
# Adaptation from Python 2.7 to Python 3, and pep8 compliance:
# Valentin Emiya, Aix-Marseille University, 2019.
# 
# Licence is granted to freely use and distribute for any sensible/legal
# purpose so long as this comment remains in any distributed code.


class GameState:
    """
    A state of the game, i.e. the game board. These are the only functions
    which are absolutely necessary to implement uct in any 2-player complete
    information deterministic zero-sum game, although they can be enhanced
    and made quicker, for example by using a GetRandomMove() function to
    generate a random move during roll-out.
    By convention the players are numbered 1 and 2.
    """
    def __init__(self):
        # At the root pretend the player just moved is player 2 - player 1
        # has the first move
        self.player_just_moved = 2
        
    def clone(self):
        """ Create a deep clone of this game state.
        """
        st = GameState()
        st.player_just_moved = self.player_just_moved
        return st

    def do_move(self, move):
        """ Update a state by carrying out the given move.
            Must update player_just_moved.
        """
        self.player_just_moved = 3 - self.player_just_moved
        
    def get_moves(self):
        """ Get all possible moves from this state.
        """
    
    def get_result(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """

    def __repr__(self):
        """ Don't need this - but good style.
        """
        pass


class NimState:
    """
    A state of the game Nim. In Nim, players alternately take 1,2 or 3 chips
    with the winner being the player to take the last chip.
    In Nim any initial state of the form 4n+k for k = 1,2,3 is a win for
    player 1 (by choosing k) chips.
    Any initial state of the form 4n is a win for player 2.
    """
    def __init__(self, ch):
        # At the root pretend the player just moved is p2 - p1 has the first
        # move
        self.player_just_moved = 2
        self.chips = ch
        
    def clone(self):
        """ Create a deep clone of this game state.
        """
        st = NimState(self.chips)
        st.player_just_moved = self.player_just_moved
        return st

    def do_move(self, move):
        """ update a state by carrying out the given move.
            Must update player_just_moved.
        """
        assert 1 <= move <= 3 and move == int(move)
        self.chips -= move
        self.player_just_moved = 3 - self.player_just_moved
        
    def get_moves(self):
        """ Get all possible moves from this state.
        """
        return list(range(1, min([4, self.chips + 1])))
    
    def get_result(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        assert self.chips == 0
        if self.player_just_moved == playerjm:
            return 1.0  # playerjm took the last chip and has won
        else:
            return 0.0  # playerjm's opponent took the last chip and has won

    def __repr__(self):
        s = "Chips:" + str(self.chips)
        s += " JustPlayed:" + str(self.player_just_moved)
        return s


class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """
    def __init__(self):
        # At the root pretend the player just moved is p2 - p1 has the first
        # move
        self.player_just_moved = 2
        # 0 = empty, 1 = player 1, 2 = player 2
        self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        
    def clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.player_just_moved = self.player_just_moved
        st.board = self.board[:]
        return st

    def do_move(self, move):
        """ update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert isinstance(move, int)
        assert 0 <= move <= 8 and not self.board[move]
        self.player_just_moved = 3 - self.player_just_moved
        self.board[move] = self.player_just_moved
        
    def get_moves(self):
        """ Get all possible moves from this state.
        """
        return [i for i in range(9) if self.board[i] == 0]
    
    def get_result(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        for (x, y, z) in [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6),
                          (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == playerjm:
                    return 1.0
                else:
                    return 0.0
        if len(self.get_moves()) == 0:
            return 0.5  # draw
        assert False  # Should not be possible to get here

    def __repr__(self):
        s = ""
        for i in range(9): 
            s += ".XO"[self.board[i]]
            if i % 3 == 2:
                s += "\n"
        return s


class OthelloState:
    """
    A state of the game of Othello, i.e. the game board.

    The board is a 2D array where 0 = empty (.), 1 = player 1 (X),
    2 = player 2 (O).
    In Othello players alternately place pieces on a square board - each piece
    played has to sandwich opponent pieces between the piece played and
    pieces already on the board. Sandwiched pieces are flipped.
    This implementation modifies the rules to allow variable sized square
    boards and terminates the game as soon as the player about to move
    cannot make a move (whereas the standard game allows for a pass move).
    """
    def __init__(self, sz=8):
        # At the root pretend the player just moved is p2 - p1 has the first
        # move
        self.player_just_moved = 2
        self.board = []  # 0 = empty, 1 = player 1, 2 = player 2
        self.size = sz
        assert sz == int(sz) and sz % 2 == 0  # size must be integral and even
        for y in range(sz):
            self.board.append([0]*sz)
        self.board[sz//2][sz//2] = self.board[sz//2-1][sz//2-1] = 1
        self.board[sz//2][sz//2-1] = self.board[sz//2-1][sz//2] = 2

    def clone(self):
        """ Create a deep clone of this game state.
        """
        st = OthelloState()
        st.player_just_moved = self.player_just_moved
        st.board = [self.board[i][:] for i in range(self.size)]
        st.size = self.size
        return st

    def do_move(self, move):
        """ update a state by carrying out the given move.
            Must update playerToMove.
        """
        (x, y) = (move[0], move[1])
        assert x == int(x) and y == int(y) and self.is_on_board(x, y) and \
            self.board[x][y] == 0
        m = self.get_all_sandwiched_counters(x, y)
        self.player_just_moved = 3 - self.player_just_moved
        self.board[x][y] = self.player_just_moved
        for (a, b) in m:
            self.board[a][b] = self.player_just_moved
    
    def get_moves(self):
        """ Get all possible moves from this state.
        """
        return [(x, y) for x in range(self.size) for y in range(self.size)
                if self.board[x][y] == 0
                and self.exists_sandwiched_counter(x, y)]

    def adjacent_to_enemy(self, x, y):
        """
        Speeds up get_moves by only considering squares which are adjacent to
        an enemy-occupied square.
        """
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1),
                         (0, -1), (-1, -1), (-1, 0), (-1, +1)]:
            if self.is_on_board(x + dx, y + dy) \
                    and self.board[x+dx][y+dy] == self.player_just_moved:
                return True
        return False
    
    def adjacent_enemy_directions(self, x, y):
        """
        Speeds up get_moves by only considering squares which are adjacent to
        an enemy-occupied square.
        """
        es = []
        for (dx, dy) in [(0, +1), (+1, +1), (+1, 0), (+1, -1),
                         (0, -1), (-1, -1), (-1, 0), (-1, +1)]:
            if self.is_on_board(x + dx, y + dy) \
                    and self.board[x+dx][y+dy] == self.player_just_moved:
                es.append((dx, dy))
        return es
    
    def exists_sandwiched_counter(self, x, y):
        """
        Does there exist at least one counter which would be flipped if my
        counter was placed at (x,y)?
        """
        for (dx, dy) in self.adjacent_enemy_directions(x, y):
            if len(self.sandwiched_counters(x, y, dx, dy)) > 0:
                return True
        return False
    
    def get_all_sandwiched_counters(self, x, y):
        """
        Is (x,y) a possible move (i.e. opponent counters are sandwiched between
         (x,y) and my counter in some direction)?
        """
        sandwiched = []
        for (dx, dy) in self.adjacent_enemy_directions(x, y):
            sandwiched.extend(self.sandwiched_counters(x, y, dx, dy))
        return sandwiched

    def sandwiched_counters(self, x, y, dx, dy):
        """
        Return the coordinates of all opponent counters sandwiched between
         (x,y) and my counter.
        """
        x += dx
        y += dy
        sandwiched = []
        while self.is_on_board(x, y) \
                and self.board[x][y] == self.player_just_moved:
            sandwiched.append((x, y))
            x += dx
            y += dy
        if self.is_on_board(x, y) \
                and self.board[x][y] == 3 - self.player_just_moved:
            return sandwiched
        else:
            return []  # nothing sandwiched

    def is_on_board(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size
    
    def get_result(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        jmcount = len([(x, y)
                       for x in range(self.size) for y in range(self.size)
                       if self.board[x][y] == playerjm])
        notjmcount = len([(x, y)
                          for x in range(self.size) for y in range(self.size)
                          if self.board[x][y] == 3 - playerjm])
        if jmcount > notjmcount:
            return 1.0
        elif notjmcount > jmcount:
            return 0.0
        else:
            return 0.5  # draw

    def __repr__(self):
        s = ""
        for y in range(self.size-1, -1, -1):
            s += ' '
            for x in range(self.size):
                s += ".XO"[self.board[x][y]]
            s += "\n"
        return s
