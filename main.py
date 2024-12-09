import numpy as np

#Nuovo bellissimo commento

class Board:

    def __init__(self, nrow: int = 7, ncol: int = 6):
        self.nrow = nrow
        self.ncol = ncol

        self.turn = 0
        self.board = np.zeros(shape=(nrow, ncol))
        # An anrray that shows content of the board

        self.column_limits = np.array(shape=ncol)

    def is_terminal():
        """
        Checks if a given board configuration is a terminal state
        """
        pass

    def legal_moves(self):
        """
        Returns an array of indices of columns where a piece can be 
        dropped
        """
        if self.is_terminal():
            return None
        return np.where(self.column_limits <= self.nrow)[0]

    def make_move():
        pass

    def minimax():
        pass

    def alphabeta():
        pass
