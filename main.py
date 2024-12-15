import numpy as np
import time

MINPLAYER: int = -1
MAXPLAYER: int = 1
EMPTY: int = 0

# Se self.has_ended è uguale a 2 allora è patta
MAP = np.array([[3, 4, 5, 7, 5, 4, 3],
                [4, 6, 8, 10, 8, 6, 4],
                [5, 7, 11, 13, 11, 7, 5],
                [5, 7, 11, 13, 11, 7, 5],
                [4, 6, 8, 10, 8, 6, 4],
                [3, 4, 5, 7, 5, 4, 3]])

# Se self.has_ended è uguale a 2 allora è patta


class Board:

    def __init__(self, nrow: int = 6, ncol: int = 7):
        """
        Initializes a new empty board
        """
        self.nrow: int = nrow
        self.ncol: int = ncol

        self.turn: int = 0
        self.has_ended: int = 0

        self.history: list[int] = []
        self.save_value_table: float = 0

        self.board = np.zeros(shape=(nrow, ncol), dtype=np.int64)
        self.column_limits = np.zeros(shape=ncol, dtype=np.int64)

    def __str__(self):
        board_str = ''
        def sym(x): return 'O' if x == MAXPLAYER else '#'
        print(" 0 1 2 3 4 5 6")
        for row in self.board[::-1]:
            row_str = '|' + '|'.join((sym(cell)) if cell !=
                                     0 else ' ' for cell in row) + '|'
            board_str += row_str + '\n'
        return board_str

    def curr_player(self):
        """
        Returns the current playing player (haha)
        """
        return MAXPLAYER if self.turn % 2 == 0 else MINPLAYER

    def curr_player_name(self):
        """
        Returns the current playing player (haha)
        """
        return "MAXPLAYER" if self.turn % 2 == 0 else "MINPLAYER"

    def legal_moves(self) -> np.ndarray:
        """
        Returns an array of indices of columns where a piece can be 
        dropped

        For the starting state for example it would be [0, 1, 2, 3, 4, 5, 6]
        """
        if self.has_ended != 0:
            return None
        else:
            return np.where(self.column_limits < self.nrow)[0]

    def make_move(self, move: int) -> None:
        """
        Updates the current board rappresentation given the move: the column in which 
        to drop the piece. This function incrementally checks if the given move ends
        the game connecting four or more.
        """
        if self.has_ended != 0:
            raise Exception("Game already ended")

        if move not in self.legal_moves():
            raise Exception("Illegal move")

        curr_player = self.curr_player()
        row, col = self.column_limits[move], move
        self.board[row, col] = curr_player
        self.column_limits[col] += 1
        self.save_value_table += MAP[row, col]*curr_player
        self.history.append(move)
        self.turn += 1

        # Verticale
        connected_points = 0
        for crow in range(row - 3, row + 4):
            if crow < self.nrow and crow >= 0:
                if self.board[crow, col] == curr_player:
                    connected_points += 1
                    if connected_points >= 4:
                        self.has_ended = curr_player
                        return
                else:
                    connected_points = 0

        # Orrizzontale
        connected_points = 0
        for ccol in range(col - 3, col + 4):
            if ccol < self.ncol and ccol >= 0:
                if self.board[row, ccol] == curr_player:
                    connected_points += 1
                    if connected_points == 4:
                        self.has_ended = curr_player
                        return
                else:
                    connected_points = 0

        # Diagonale basso_sinistra -> alto_destra
        connected_points = 0

        #!! 2 CICLI FOR SONO DA TOGLIERE, MA AL MOMENTO NON SO QUALI

        for d in range(-3, 4):
            if col + d < self.ncol and row + d >= 0 and row + d < self.nrow and col + d >= 0:
                if self.board[row + d, col + d] == curr_player:
                    connected_points += 1
                    if connected_points >= 4:
                        self.has_ended = curr_player
                        return
                else:
                    connected_points = 0

        # Diagonale alto_sinistra -> basso_destra
        connected_points = 0
        for d in range(-3, 4):
            if row + d < self.nrow and col - d >= 0 and col - d < self.ncol and row + d >= 0:
                if self.board[row + d, col - d] == curr_player:
                    connected_points += 1
                    if connected_points >= 4:
                        self.has_ended = curr_player
                        return
                else:
                    connected_points = 0

        if self.legal_moves() is None:
            self.has_ended = 2

    def undo_move(self) -> None:
        """
        Undoes the last move
        """
        last_move = self.history.pop()
        edit_row, edit_col = self.column_limits[last_move] - 1, last_move

        self.save_value_table -= MAP[edit_row,
                                     edit_col]*self.board[edit_row, edit_col]
        # Remove the disc from the board
        self.board[edit_row, edit_col] = EMPTY
        # Decrease the column height
        self.column_limits[edit_col] -= 1
        # Set the correct turn
        self.turn -= 1
        # Restore the previous situation
        self.has_ended = EMPTY

    # Heuristics

    def eval_position(self) -> float:
        """
        Uses a precalculated value that is updated with each move.
        Closely related to the nummber of ways one can win in that square.
        """
        if self.has_ended:
            return float("+inf")*self.has_ended
        return self.save_value_table

    def eval_possibilities(self) -> float:
        """
        Allora, più o meno controllo ogni possibile mossa dove formo una riga e se quello dopo è una 
        casella vuota allora va bene, altrimenti non va bene. 
        Questo solamente per quando posso formare 3 o 2, non mi importa quando formo 1. 


        Se formo 4, ovviamente ritorno infinito*curr_player
        Quindi se vedo 3 in fila e quello prima o quello dopo sono liberi, è un'ottima posizione


        Se vedo 2 in fila la situazione diventa più spinosa, perché devo controllare o 2 avanti, 
        o 2 indietro o 1 avanti e 1 indietro.


        Conto che sia meglio portare me in una situazione vantaggiosa che mettere in difficoltà 
        l'avversario. In futuro potrei fare una versione che controlla anche tra quante mosse potrei 
        arrivare alla configurazione desiderata, dove se è 2 non conviene giocare la mossa di solito.

        Adesso mi manca da pensare: controllo per ogni mossa mia anche quella avversaria o no? 
        O semplicemente conto come se l'avversario stia per giocare ma ci 
        tolgo 1 (/lo setto a 3 se >= 4) perché non è ancora così pericoloso?

        """
        curr_player = self.curr_player()
        if self.legal_moves() is None or len(self.legal_moves()) == 0:
            return 0
        values = [0, 0, 1, 3]
        adv_values = [0, 0, 0, 0.5, 2, 2, 2, 2, 2, 2, 2, 2]
        tot = 0
        for r in self.legal_moves():
            row, col = self.column_limits[r], r
            self.board[row, col] = curr_player
            self.column_limits[col] += 1
            # Controlli :(
            counter = 0
            for rrow in range(row - 3, row + 4):
                if rrow >= 0 and rrow < self.nrow:
                    if counter == 4:
                        self.board[row, col] = EMPTY
                        self.column_limits[col] -= 1
                        return float("+inf")*curr_player
                    if self.board[rrow, col] != curr_player:
                        if rrow - counter - 1 >= 0:
                            if self.board[rrow - counter - 1, col] == 0:
                                if counter == 3:
                                    tot += values[counter]*curr_player
                                elif rrow - counter - 2 >= 0:
                                    if self.board[rrow - counter - 2, col] == 0:
                                        tot += values[counter]*curr_player
                                if self.board[rrow, col] == 0 and counter == 2:
                                    tot += values[counter]*curr_player
                        if self.board[rrow, col] == 0:
                            if counter == 3:
                                tot += values[counter]*curr_player
                            elif rrow + 1 < self.ncol:
                                if self.board[rrow, col] == 0:
                                    tot += values[counter]*curr_player
                        counter = 0
                    else:
                        counter += 1
                elif rrow - counter - 1 >= 0 and rrow < self.nrow:
                    if self.board[rrow - counter - 1, col] == 0:
                        if counter == 3:
                            tot += values[counter]*curr_player
                        elif rrow - counter - 2 >= 0:
                            if self.board[rrow - counter - 2, col] == 0:
                                tot += values[counter]*curr_player

            counter = 0
            for ccol in range(col - 3, col + 4):
                if ccol >= 0 and ccol < self.ncol:
                    if counter == 4:
                        self.board[row, col] = EMPTY
                        self.column_limits[col] -= 1
                        return float("+inf")*curr_player
                    if self.board[row, ccol] != curr_player:
                        if ccol - counter - 1 >= 0:
                            if self.board[row, ccol - counter - 1] == 0:
                                if counter == 3:
                                    tot += values[counter]*curr_player
                                elif ccol - counter - 2 >= 0:
                                    if self.board[row, ccol - counter - 2] == 0:
                                        tot += values[counter]*curr_player
                                if self.board[row, ccol] == 0 and counter == 2:
                                    tot += values[counter]*curr_player
                        if self.board[row, ccol] == 0:
                            if counter == 3:
                                tot += values[counter]*curr_player
                            elif ccol + 1 < self.ncol:
                                if self.board[row, ccol + 1] == 0:
                                    tot += values[counter]*curr_player
                        counter = 0
                    else:
                        counter += 1
                elif ccol - counter - 1 >= 0 and ccol < self.ncol:
                    if self.board[row, ccol - counter - 1] == 0:
                        if counter == 3:
                            tot += values[counter]*curr_player
                        elif ccol - counter - 2 >= 0:
                            if self.board[row, ccol - counter - 2] == 0:
                                tot += values[counter]*curr_player

            counter = 0
            for d in range(-3, 4):
                if row + d >= 0 and row + d < self.nrow and col + d >= 0 and col + d < self.ncol:
                    if counter == 4:
                        self.board[row, col] = EMPTY
                        self.column_limits[col] -= 1
                        return float("+inf")*curr_player
                    if self.board[row + d, col + d] != curr_player:
                        if row + d - counter - 1 >= 0 and col + d - counter - 1 >= 0:
                            if self.board[row + d - counter - 1, col + d - counter - 1] == 0:
                                if counter == 3:
                                    tot += values[counter]*curr_player
                                elif row + d - counter - 2 >= 0 and col + d - counter - 2 >= 0:
                                    if self.board[row + d - counter - 2, col + d - counter - 2] == 0:
                                        tot += values[counter]*curr_player
                                if self.board[row + d, col + d] == 0 and counter == 2:
                                    tot += values[counter]*curr_player
                    else:
                        counter += 1
                elif row + d - counter - 1 >= 0 and col + d - counter - 1 >= 0 and row + d < self.nrow and col + d < self.ncol:
                    if self.board[row + d - counter - 1, col + d - counter - 1] == 0:
                        if counter == 3:
                            tot += values[counter]*curr_player
                        elif row + d - counter - 2 >= 0 and col + d - counter - 2 >= 0:
                            if self.board[row + d - counter - 2, col + d - counter - 2] == 0:
                                tot += values[counter]*curr_player

            counter = 0
            for d in range(-3, 4):
                if row + d >= 0 and row + d < self.nrow and col - d >= 0 and col - d < self.ncol:
                    if counter == 4:
                        self.board[row, col] = EMPTY
                        self.column_limits[col] -= 1
                        return float("+inf")*curr_player
                    if self.board[row + d, col - d] != curr_player:
                        if row + d - counter - 1 >= 0 and col - d - counter - 1 >= 0:
                            if self.board[row + d - counter - 1, col - d - counter - 1] == 0:
                                if counter == 3:
                                    tot += values[counter]*curr_player
                                elif row + d - counter - 2 >= 0 and col - d - counter - 2 >= 0:
                                    if self.board[row + d - counter - 2, col - d - counter - 2] == 0:
                                        tot += values[counter]*curr_player
                                if self.board[row + d, col - d] == 0 and counter == 2:
                                    tot += values[counter]*curr_player
                    else:
                        counter += 1
                elif row + d - counter - 1 >= 0 and col - d - counter - 1 >= 0 and row + d < self.nrow and col - d < self.ncol:
                    if self.board[row + d - counter - 1, col - d - counter - 1] == 0:
                        if counter == 3:
                            tot += values[counter]*curr_player
                        elif row + d - counter - 2 >= 0 and col - d - counter - 2 >= 0:
                            if self.board[row + d - counter - 2, col - d - counter - 2] == 0:
                                tot += values[counter]*curr_player

            self.board[row, col] = EMPTY
            self.column_limits[col] -= 1

        curr_player = -curr_player
        for r in self.legal_moves():
            row, col = self.column_limits[r], r
            self.board[row, col] = curr_player
            self.column_limits[col] += 1
            # Controlli :(
            counter = 0
            for rrow in range(row - 3, row + 4):
                if rrow >= 0 and rrow < self.nrow:
                    if counter == 4:
                        tot -= adv_values[counter]*curr_player
                    if self.board[rrow, col] != curr_player:
                        if rrow - counter - 1 >= 0:
                            if self.board[rrow - counter - 1, col] == 0:
                                if counter == 3:
                                    tot -= adv_values[counter]*curr_player
                                elif rrow - counter - 2 >= 0:
                                    if self.board[rrow - counter - 2, col] == 0:
                                        tot -= adv_values[counter]*curr_player
                                if self.board[rrow, col] == 0 and counter == 2:
                                    tot -= adv_values[counter]*curr_player
                        if self.board[rrow, col] == 0:
                            if counter == 3:
                                tot -= adv_values[counter]*curr_player
                            elif rrow + 1 < self.ncol:
                                if self.board[rrow, col] == 0:
                                    tot -= adv_values[counter]*curr_player
                        counter = 0
                    else:
                        counter += 1
                elif rrow - counter - 1 >= 0 and rrow < self.nrow:
                    if self.board[rrow - counter - 1, col] == 0:
                        if counter == 3:
                            tot -= adv_values[counter]*curr_player
                        elif rrow - counter - 2 >= 0:
                            if self.board[rrow - counter - 2, col] == 0:
                                tot -= adv_values[counter]*curr_player

            counter = 0
            for ccol in range(col - 3, col + 4):
                if ccol >= 0 and ccol < self.ncol:
                    if counter == 4:
                        tot -= adv_values[counter]*curr_player
                    if self.board[row, ccol] != curr_player:
                        if ccol - counter - 1 >= 0:
                            if self.board[row, ccol - counter - 1] == 0:
                                if counter == 3:
                                    tot -= adv_values[counter]*curr_player
                                elif ccol - counter - 2 >= 0:
                                    if self.board[row, ccol - counter - 2] == 0:
                                        tot -= adv_values[counter]*curr_player
                                if self.board[row, ccol] == 0 and counter == 2:
                                    tot -= adv_values[counter]*curr_player
                        if self.board[row, ccol] == 0:
                            if counter == 3:
                                tot -= adv_values[counter]*curr_player
                            elif ccol + 1 < self.ncol:
                                if self.board[row, ccol + 1] == 0:
                                    tot -= adv_values[counter]*curr_player
                        counter = 0
                    else:
                        counter += 1
                elif ccol - counter - 1 >= 0 and ccol < self.ncol:
                    if self.board[row, ccol - counter - 1] == 0:
                        if counter == 3:
                            tot -= adv_values[counter]*curr_player
                        elif ccol - counter - 2 >= 0:
                            if self.board[row, ccol - counter - 2] == 0:
                                tot -= adv_values[counter]*curr_player

            counter = 0
            for d in range(-3, 4):
                if row + d >= 0 and row + d < self.nrow and col + d >= 0 and col + d < self.ncol:
                    if counter == 4:
                        tot -= adv_values[counter]*curr_player
                    if self.board[row + d, col + d] != curr_player:
                        if row + d - counter - 1 >= 0 and col + d - counter - 1 >= 0:
                            if self.board[row + d - counter - 1, col + d - counter - 1] == 0:
                                if counter == 3:
                                    tot -= adv_values[counter]*curr_player
                                elif row + d - counter - 2 >= 0 and col + d - counter - 2 >= 0:
                                    if self.board[row + d - counter - 2, col + d - counter - 2] == 0:
                                        tot -= adv_values[counter]*curr_player
                                if self.board[row + d, col + d] == 0 and counter == 2:
                                    tot -= adv_values[counter]*curr_player
                    else:
                        counter += 1
                elif row + d - counter - 1 >= 0 and col + d - counter - 1 >= 0 and row + d < self.nrow and col + d < self.ncol:
                    if self.board[row + d - counter - 1, col + d - counter - 1] == 0:
                        if counter == 3:
                            tot -= adv_values[counter]*curr_player
                        elif row + d - counter - 2 >= 0 and col + d - counter - 2 >= 0:
                            if self.board[row + d - counter - 2, col + d - counter - 2] == 0:
                                tot -= adv_values[counter]*curr_player

            counter = 0
            for d in range(-3, 4):
                if row + d >= 0 and row + d < self.nrow and col - d >= 0 and col - d < self.ncol:
                    if counter == 4:
                        tot -= adv_values[counter]*curr_player
                    if self.board[row + d, col - d] != curr_player:
                        if row + d - counter - 1 >= 0 and col - d - counter - 1 >= 0:
                            if self.board[row + d - counter - 1, col - d - counter - 1] == 0:
                                if counter == 3:
                                    tot -= adv_values[counter]*curr_player
                                elif row + d - counter - 2 >= 0 and col - d - counter - 2 >= 0:
                                    if self.board[row + d - counter - 2, col - d - counter - 2] == 0:
                                        tot -= adv_values[counter]*curr_player
                                if self.board[row + d, col - d] == 0 and counter == 2:
                                    tot -= adv_values[counter]*curr_player
                    else:
                        counter += 1
                elif row + d - counter - 1 >= 0 and col - d - counter - 1 >= 0 and row + d < self.nrow and col - d < self.ncol:
                    if self.board[row + d - counter - 1, col - d - counter - 1] == 0:
                        if counter == 3:
                            tot -= adv_values[counter]*curr_player
                        elif row + d - counter - 2 >= 0 and col - d - counter - 2 >= 0:
                            if self.board[row + d - counter - 2, col - d - counter - 2] == 0:
                                tot -= adv_values[counter]*curr_player

            self.board[row, col] = EMPTY
            self.column_limits[col] -= 1
        return tot

    def connections_eval(self) -> int:
        """
        This function computes the number of possible 4 in a row that each player can
        still make and returns the difference
        added: if one of these windows contains two or three pieces, add a bonus
        """

        def count_open_sequences(player):
            count = 0
            # Horizontal
            for r in range(self.nrow):
                for c in range(self.ncol - 3):
                    window = self.board[r, c:c+4]
                    if is_valid_window(window, player):
                        count += relative_worth(window,
                                                player, (r, c), (r, c+3))
            # Vertical
            for r in range(self.nrow - 3):
                for c in range(self.ncol):
                    window = self.board[r:r+4, c]
                    if is_valid_window(window, player):
                        count += relative_worth(window,
                                                player, (r, c), (r+3, c))
            # Diagonal (top-left to bottom-right)
            for r in range(self.nrow - 3):
                for c in range(self.ncol - 3):
                    window = [self.board[r+i, c+i] for i in range(4)]
                    if is_valid_window(window, player):
                        count += relative_worth(window,
                                                player, (r, c), (r+3, c+3))
            # Diagonal (top-right to bottom-left)
            for r in range(self.nrow - 3):
                for c in range(3, self.ncol):
                    window = [self.board[r+i, c-i] for i in range(4)]
                    if is_valid_window(window, player):
                        count += relative_worth(window,
                                                player, (r, c), (r+3, c-3))
            return count

        def is_valid_window(window, player):
            """
            Checks if a window contains only the player's pieces and empty spaces.
            """
            return all(cell == player or cell == 0 for cell in window)

        def value_of_window(window, player):
            """
            Check if a window contains a 'winning' ammount of tiles for a player:
            - any opponent tile in the window: not a usable window (0)
            - 1 tile and 3 empty spaces: not much to say (1)
            - 2 tiles and 2 empty spaces: good window (3)
            - 3 tiles and 1 empty space: very good window (8)
            - 4 tiles: winning position (10000)
            """
            res = []
            for cell in window:
                if cell == -player:
                    return 0
                elif cell == 0:
                    res.append(0)
                elif cell == player:
                    res.append(1)
            res = sum(res)
            if res == 0:
                return 0
            elif res == 1:
                return 1
            elif res == 2:
                return 3
            elif res == 3:
                return 8
            elif res == 4:
                return 10000

        def relative_worth(window, player, start, end):
            """
            Function that comutes the worth of a window and relates it
            to how hard it is to fill it (how 'high' its empty positions are)
            'start' and 'end' are the coordinates of the extremes of the window (used to locate it)
            """
            res = value_of_window(window, player)
            if res == 0:  # End if window has no value
                return res
            # Compute indexes of squares in the window to later evaluate how 'high' empty squares are
            rows = np.linspace(start[0], end[0], 4) if start[0] != end[0] else [
                start[0] for _ in range(4)]
            cols = np.linspace(start[1], end[1], 4) if start[1] != end[1] else [
                start[1] for _ in range(4)]
            rows, cols = [int(r) for r in rows], [int(c) for c in cols]

            # Count how many tiles have been placed on each column
            bottom = [self.history.count(j) for j in range(self.ncol)]
            # For each column that contains a zero, compute how far we are from the bottom
            depth = []
            for col, row in zip(cols, rows):
                if self.board[row, col] == 0:
                    # row-1 for later computations (can i place a tile in the window right now?)
                    depth.append((row-1) - bottom[col])
            # If the window is 'far' from completition, assign less value
            # Total of distances from being able to place a tile in the window
            depth = sum(depth)
            # Decay factor of 0.9 the further the winning window is
            return res * (0.9)**depth

        # Count potential sequences for each player
        max_count = count_open_sequences(1)
        min_count = count_open_sequences(-1)

        # Return the difference
        return max_count - min_count

    def threats_eval(self) -> int:
        """
        Evaluation function that uses the concept of threats:
        A threat is a position that would complete a 4-in-a-row for a player.
        """
        def find_threats(player):
            """
            Identify all threats for the given player.
            """
            threats = set()
            # Horizontal threats
            for r in range(self.nrow):
                for c in range(self.ncol - 3):
                    window = self.board[r, c:c+4]
                    if is_threat_window(window, player):
                        for i in range(4):
                            if window[i] == 0:  # Empty spot in the window
                                threats.add((r, c + i))
            # Vertical threats
            for r in range(self.nrow - 3):
                for c in range(self.ncol):
                    window = self.board[r:r+4, c]
                    if is_threat_window(window, player):
                        for i in range(4):
                            if window[i] == 0:  # Empty spot in the window
                                threats.add((r + i, c))
            # Diagonal (bottom-left to top-right) threats
            for r in range(self.nrow - 3):
                for c in range(self.ncol - 3):
                    window = [self.board[r+i, c+i] for i in range(4)]
                    if is_threat_window(window, player):
                        for i in range(4):
                            if window[i] == 0:  # Empty spot in the window
                                threats.add((r + i, c + i))
            # Diagonal (top-left to bottom-right) threats
            for r in range(3, self.nrow):
                for c in range(self.ncol - 3):
                    window = [self.board[r-i, c+i] for i in range(4)]
                    if is_threat_window(window, player):
                        for i in range(4):
                            if window[i] == 0:  # Empty spot in the window
                                threats.add((r + i, c - i))
            return threats

        def is_threat_window(window, player):
            """
            A window is a threat if it contains exactly 3 pieces of the player and 1 empty space.
            """
            return np.count_nonzero(window == player) == 3 and np.count_nonzero(window == 0) == 1

        def filter_threats(threats, opponent_threats):
            """
            Remove useless threats:
            - Threats directly above opponent threats.
            - Threats above threats shared by both players.
            """
            filtered = set()
            for r, c in threats:
                if (r + 1, c) not in opponent_threats or r + 1 >= self.nrow:
                    filtered.add((r, c))
            return filtered

        def score_threats(threats):
            """
            Assign scores to threats:
            - Base score for each threat.
            - Bonus for the lowest threat in a column.
            - Bonus for consecutive threats by the same player.
            """
            score = 0
            column_bottoms = {c: self.nrow for c in range(self.ncol)}
            consecutive_bonus = 2

            # Determine the lowest threat in each column
            for r, c in threats:
                column_bottoms[c] = min(column_bottoms[c], r)

            for r, c in threats:
                # Base score
                score += 1
                # Bonus for being the lowest in a column
                if r == column_bottoms[c]:
                    score += 2
                # Bonus for consecutive threats (check neighbors)
                if (r, c + 1) in threats or (r + 1, c) in threats:
                    score += consecutive_bonus
            return score

        # Find threats for both players
        max_threats = find_threats(1)
        min_threats = find_threats(-1)

        # Filter threats
        max_threats = filter_threats(max_threats, min_threats)
        min_threats = filter_threats(min_threats, max_threats)

        # Calculate scores
        max_score = score_threats(max_threats)
        min_score = score_threats(min_threats)

        # Return the difference in threat scores
        return max_score - min_score

    def eval_brullen(self):
        return self.threats_eval() + self.connections_eval()

    def eval_simple(self) -> float:
        if self.has_ended != 2:
            return self.has_ended
        else:
            return 0

    def eval_cartago(self) -> float:
        if self.has_ended == 1:
            return float("inf")
        elif self.has_ended == -1:
            return float("-inf")
        elif self.has_ended == 2:
            return 0
        else:
            # Ritorno quante connessioni ha fatto l'avversario nell'ultimo intorno della mossa
            mboard = self.board * [1, 1.5, 2, 2.5, 2, 1.5, 1]
            # Diagonal sums
            diagonal_sums = []
            for offset in range(-self.nrow + 1, self.ncol):
                diagonal = np.diagonal(mboard, offset=offset)
                diagonal_sums.append(np.sum(diagonal))
            # Antidiagonal sums
            antidiagonal_sums = []
            flipped = np.fliplr(mboard)
            for offset in range(-self.nrow + 1, self.ncol):
                diagonal = np.diagonal(flipped, offset=offset)
                antidiagonal_sums.append(np.sum(diagonal))
            # Row sum
            rscore = mboard.sum(axis=0)
            # Column sum
            cscore = mboard.sum(axis=1)
            # Combination
            score = (np.max(cscore) + np.min(cscore) +
                     np.max(rscore) + np.min(rscore) +
                     np.max(diagonal_sums) + np.min(diagonal_sums) +
                     np.max(antidiagonal_sums) + np.min(antidiagonal_sums))

            return score

    # Tree search

    def minimax(self, depth) -> tuple[int, float]:
        """
        Minimax!
        """

        if self.has_ended == 1 or self.has_ended == -1 or depth <= 0 or len(self.legal_moves()) == 0:
            return self.history[-1], self.eval_possibilities() + self.eval_Chiorri()

        if self.legal_moves() is None:
            return self.history[-1], 0

        curr_pl = self.curr_player()
        best = self.legal_moves()[3]

        if curr_pl == MAXPLAYER:
            a = float("-inf")
            for move in self.legal_moves():
                self.make_move(move)
                _, value = self.minimax(depth - 1)
                if value > a:
                    a = value
                    best = move
                self.undo_move()
        else:
            a = float('+inf')
            for move in self.legal_moves():
                self.make_move(move)
                _, value = self.minimax(depth - 1)
                if value < a:
                    a = value
                    best = move
                self.undo_move()
        return best, a

    def alphabeta(self,
                  depth,
                  alpha=float('-inf'),
                  beta=float('inf'),
                  evaluation=(lambda x: 0)
                  ) -> tuple[int, float]:
        """
        Minimax with alphabeta pruning!
        """

        if self.has_ended == 1 or self.has_ended == -1 or depth <= 0:
            return self.history[-1], evaluation(self)

        moves = self.legal_moves()

        if moves is None or len(moves) == 0:
            return self.history[-1], 0

        curr_pl = self.curr_player()
        best = moves[0]

        if curr_pl == MAXPLAYER:
            value = float('-inf')
            for move in moves:
                self.make_move(move)
                _, score = self.alphabeta(
                    depth - 1, alpha, beta, evaluation=evaluation)
                self.undo_move()
                if score > value:
                    value = score
                    best = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float('inf')
            for move in moves:
                self.make_move(move)
                _, score = self.alphabeta(
                    depth - 1, alpha, beta, evaluation=evaluation)
                self.undo_move()
                if score < value:
                    value = score
                    best = move
                beta = min(beta, value)
                if alpha >= beta:
                    break

        return best, value

    def gen_move(self,
                 base_depth: int = 8,
                 eur=(lambda x: x.eval_position())):
        """
        Returns the move chosen by alphabeta pruning. 
        """

        depth = (base_depth
                 if self.turn <= 10
                 else min(base_depth + int((self.turn - 10)/2), 30)
                 )

        move, _ = self.alphabeta(
            depth,
            evaluation=eur
        )

        return move


if __name__ == "__main__":

    game_board = Board()
    player = str(input("Select player (MAX or MIN): "))
    while player != "MAX" and player != "MIN" and player != "max" and player != "min":
        player = str(input("Select player (MAX or MIN): "))

    player = MAXPLAYER if player == "MAX" or player == "max" else MINPLAYER

    while game_board.has_ended == 0:

        print(f"Move({game_board.turn}) Plays: {
              game_board.curr_player_name()}\n")

        if game_board.curr_player() == player:
            print(game_board, "\n")
            lm = game_board.legal_moves()
            move = int(input(f"{lm}> "))
            while move not in lm:
                move = input(f"{lm}> ")
            game_board.make_move(move)
        else:
            start = time.time()
            mossa = game_board.gen_move(base_depth=8)
            game_board.make_move(mossa)
            end = time.time()
            # print(f"Elapsed: {end - start}")

    print(game_board)

    print()
    print("MAX WON" if game_board.has_ended == 1 else (
        "DRAW"if game_board.has_ended == 2 or game_board.has_ended == 0 else "MIN WON"))
