import numpy as np

MINPLAYER: int = -1
MAXPLAYER: int = 1
EMPTY: int = 0


class Board:

    def __init__(self, nrow: int = 6, ncol: int = 7):
        """
        Initializes a new empty board
        """
        self.nrow: int = nrow
        self.ncol: int = ncol

        self.turn: int = 0
        self.has_ended: int = False
        self.history: list[int] = []
        self.board = np.zeros(shape=(nrow, ncol))
        self.column_limits = np.zeros(shape=ncol)

    def __str__(self):
        board_str = ''
        def sym(x): return '○' if sym == MAXPLAYER else '●'
        for row in self.board:
            row_str = ('|'.join(sym(cell)) if cell !=
                       0 else ' ' for cell in row)
            board_str += row_str + '\n'
        return board_str

    def curr_player(self):
        """
        Returns the current playing player (haha)
        """
        return MAXPLAYER if self.turn % 2 else MINPLAYER

    def is_terminal(self) -> int:
        """
        Checks if a given board configuration is a terminal state

        Returns:
            - +1  MINPLAYER if the board configuration is a win for MINPLAYER
            - -1  MAXPLAYER if the board configuration is a win for MAXPLAYER
            -  0  if the game is not over yet
        """
        for i in range(self.nrow - 3):
            for j in range(self.ncol - 3):
                if self.board[i, j] == 0:
                    continue
                if self.board[i, j] == MINPLAYER and self.board[i, j + 1] == MINPLAYER and self.board[i, j + 2] == MINPLAYER and self.board[i, j + 3] == MINPLAYER:
                    return MINPLAYER
                elif self.board[i, j] == MINPLAYER and self.board[i + 1, j + 1] == MINPLAYER and self.board[i + 2, j + 2] == MINPLAYER and self.board[i + 3, j + 3] == MINPLAYER:
                    return MINPLAYER
                elif self.board[i, j] == MINPLAYER and self.board[i + 1, j] == MINPLAYER and self.board[i + 2, j] == MINPLAYER and self.board[i + 3, j] == MINPLAYER:
                    return MINPLAYER
                elif self.board[i + 3, j] == MINPLAYER and self.board[i + 2, j + 1] == MINPLAYER and self.board[i + 1, j + 2] == MINPLAYER and self.board[i, j + 3] == MINPLAYER:
                    return MINPLAYER
                elif self.board[i, j] == MAXPLAYER and self.board[i, j + 1] == MAXPLAYER and self.board[i, j + 2] == MAXPLAYER and self.board[i, j + 3] == MAXPLAYER:
                    return MAXPLAYER
                elif self.board[i, j] == MAXPLAYER and self.board[i + 1, j + 1] == MAXPLAYER and self.board[i + 2, j + 2] == MAXPLAYER and self.board[i + 3, j + 3] == MAXPLAYER:
                    return MAXPLAYER
                elif self.board[i, j] == MAXPLAYER and self.board[i + 1, j] == MAXPLAYER and self.board[i + 2, j] == MAXPLAYER and self.board[i + 3, j] == MAXPLAYER:
                    return MAXPLAYER
                elif self.board[i + 3, j] == MAXPLAYER and self.board[i + 2, j + 1] == MAXPLAYER and self.board[i + 1, j + 2] == MAXPLAYER and self.board[i, j + 3] == MAXPLAYER:
                    return MAXPLAYER

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
        """
        Updates the current board rappresentation given the move: the column in which 
        to drop the piece. This function incrementally checks if the given move ends the game
        connecting four or more.
        """

        if self.is_terminal() != 0:
            raise Exception("Game already ended")

        if move not in self.legal_moves():
            raise Exception("Illegal move :(")

        curr_player = self.curr_player()
        row, col = self.column_limits[move], move

        self.board[row, col] = curr_player
        self.column_limits[col] += 1
        self.history.append(move)
        self.turn += 1

        # TODO
        # To check if the game has ended and update accordingly the board state

        connected_points = 0
        for crow in range(row - 4, row + 4 + 1):
            if crow < self.nrow and crow >= 0:
                if self.board[crow, col] == curr_player:
                    connected_points += 1
                else:
                    connected_points = 0

        if connected_points >= 4:
            self.has_ended = curr_player

        connected_points = 0
        for ccol in range(col - 4, col + 4 + 1):
            if col < self.ncol and ccol >= 0:
                if self.board[row, ccol] == curr_player:
                    connected_points += 1
                else:
                    connected_points = 0

        if connected_points >= 4:
            self.has_ended = curr_player

        connected_points = 0
        for d in range(0, 4 + 1):
            if col + d < self.ncol and row - d >= 0:
                if self.board[row - d, col + d] == curr_player:
                    connected_points += 1
                else:
                    connected_points = 0

        if connected_points >= 4:
            self.has_ended = curr_player

        connected_points = 0
        for d in range(0, 4 + 1):
            if col + d < self.nrow and col - d >= 0:
                if self.board[row + d, col - d] == curr_player:
                    connected_points += 1
                else:
                    connected_points = 0

        if connected_points >= 4:
            self.has_ended = curr_player

        # Rendere anche la generazione delle mosse di forza quattro incementale
        # ogni volta prendo il max() della sequenza di vicini più lunga delle teste
        # che "faccio crescere" a quel punto mi basta controllare se il max(...) locale
        # arriva a 4.

        # Probabilmente ha senso farlo solo in versioni generalizzate del gioco in cui ho
        # sequenze arbitrarie da controllare

    def play_move_sequence(self, move_list: list[int]) -> None:
        """
        Plays a sequence of moves on the board
        """
        for move in move_list:
            self.make_move(move)

    def undo_move_sequence(self, move_num: int) -> None:
        """
        Undoes a sequence of moves on the board
        """
        for _ in range(move_num):
            self.undo_move()

    def undo_move(self) -> None:
        """
        Undoes the last move
        """
        last_move = self.history.pop()
        edit_row, edit_col = self.column_limits[last_move] - 1, last_move

        # Remove the disc from the board
        self.board[edit_row, edit_col] = EMPTY
        # Decrease the column height
        self.column_limits[edit_col] -= 1
        # Set the correct turn
        self.turn -= 1
        # Restore the previous situation
        self.has_ended = EMPTY

    def minimax(self) -> tuple[int, float]:
        pass

    def alphabeta(self) -> tuple[int, float]:
        pass
