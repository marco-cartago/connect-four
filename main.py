import numpy as np

# Nuovo bellissimo commento

MINPLAYER: int = -1
MAXPLAYER: int = 1


class Board:

    def __init__(self, nrow: int = 7, ncol: int = 6):
        self.nrow = nrow
        self.ncol = ncol

        self.turn = 0
        self.board = np.zeros(shape=(nrow, ncol))
        # An anrray that shows content of the board

        self.column_limits = np.zeros(shape=ncol)

    def is_terminal(self) -> int:
        """
        Checks if a given board configuration is a terminal state
        """
        for i in range(self.nrow - 3):
            for j in range(self.ncol - 3):
                if self.board[i, j] == MINPLAYER and self.board[i, j + 1] == MINPLAYER and self.board[i, j + 2] == MINPLAYER and self.board[i, j + 3] == MINPLAYER:
                    return -1
        pass

    def legal_moves(self) -> np.ndarray:
        """
        Returns an array of indices of columns where a piece can be 
        dropped
        """
        if self.is_terminal():
            return None
        return np.where(self.column_limits <= self.nrow)[0]

    def make_move(self, move: int) -> None:
        pass

    def minimax(self) -> tuple[int, float]:
        pass

    def alphabeta(self) -> tuple[int, float]:
        pass
