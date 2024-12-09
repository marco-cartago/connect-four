import numpy as np

# Nuovo bellissimo commento

MINPLAYER: int = -1
MAXPLAYER: int = 1


class Board:

    def __init__(self, nrow: int = 7, ncol: int = 6):
        self.nrow = nrow
        self.ncol = ncol

        self.turn = 0
        self.history = []
        self.board = np.zeros(shape=(nrow, ncol))
        self.column_limits = np.zeros(shape=ncol)

    def curr_player(self):
        return MAXPLAYER if self.turn % 2 else MINPLAYER

    def is_terminal(self) -> int:
        """
        Checks if a given board configuration is a terminal state

        Returns:
            - +1  MINPLAYER if the board configuration is a win for MINPLAYER
            - -1  MAXPLAYER if the board configuration is a win for MAXPLAYER
            -  0  if the game is not over jet
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

        For the starting state for example it would be [0, 1, 2, 3, 4, 5, 6]

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
