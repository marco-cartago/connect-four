import numpy as np
import time

MINPLAYER: int = -1
MAXPLAYER: int = 1
EMPTY: int = 0

DEBUG_DEPTH: int = 5

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
        self.board = np.zeros(shape=(nrow, ncol), dtype=np.int64)
        self.column_limits = np.zeros(shape=ncol, dtype=np.int64)

    def __str__(self):
        board_str = ''
        def sym(x): return '○' if x == MAXPLAYER else '●'
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
            s = f'Mossa illegale! Il giocatore {self.curr_player()} ha provato a giocare {
                move} nella posizione\n{str(self)}'
            raise Exception(s)

        curr_player = self.curr_player()
        row, col = self.column_limits[move], move
        self.board[row, col] = curr_player
        self.column_limits[col] += 1
        self.history.append(move)
        self.turn += 1

        # Verticale
        connected_points = 0
        for crow in range(row - 3, row + 1):
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
        for d in range(-3, 4):
            if col + d < self.ncol and row + d < self.nrow and row + d >= 0 and col + d >= 0:
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
            if row - d < self.nrow and col + d >= 0 and col + d < self.ncol and row - d > 0:
                if self.board[row - d, col + d] == curr_player:
                    connected_points += 1
                    if connected_points >= 4:
                        self.has_ended = curr_player
                        return
                else:
                    connected_points = 0

        if self.legal_moves() is None:
            self.has_ended = 2

    def make_move_sequence(self, move_list: list[int], verbose: bool = False) -> None:
        """
        Plays a sequence of moves on the board
        """
        for move in move_list:
            if move in self.legal_moves():
                if verbose:
                    print(self)
                    for m in self.legal_moves():
                        self.make_move(m)
                        print(f"move {m} -> {self.has_ended}")
                        self.undo_move()
                self.make_move(move)

                if self.has_ended != 0:
                    break
            else:
                raise Exception(f"Illegl move {move}")

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
        self.has_ended = 0

    def eval(self) -> int:
        # Una posizione è forte quando:
        # calcolo il numero di elementi in riga in base alle caselle vuote che ha vicino
        '''
        | | | | | | | |
        |x|x|x| | | | |
        |x|○|x| |x|x|x|
        |x|●|x|x|x|○|x|
        |x|○|○|x|x|○|x|
        |●|●|○|●|●|●|●|
        In poche parole, se questa è la posizione (so che non può accadere ma è per fare un esempio)
        Io pensavo di controllare tutte le posizioni segnate con delle x
        '''
        curr_player = self.curr_player()
        tot = 0
        # Probabilmente farò in parte controllo verticale in quanto molto più veloce
        for i in range(self.ncol - 1):
            # Dove controllare
            if self.column_limits[i + 1] > self.column_limits[i]:
                # Ciclo per ogni casella da controllare
                for j in range(self.column_limits[i], self.column_limits[i + 1] + 1):
                    tmp = 0
                    tmp_tot = 0
                    # Controllo di quanti orizzontali già in linea
                    for k in range(-3, 4):
                        if k == 0:
                            continue
                        if i + k >= 0 and i + k < self.ncol:
                            if self.board[j, i + k] == EMPTY:
                                # Se il valore assoluto di tmp è maggiore del valore assoluto di tot, allora scambia
                                # Forse questo posso modificare anche in base al turno del giocatore
                                if abs(tmp) > abs(tmp_tot):
                                    # Probabilmente devo controllare a chi tocca e devo anche pensare al numero di mosse che devo fare per arrivare fin là, ma per ora va bene così
                                    tmp_tot = tmp
                                tmp = 0
                            else:
                                tmp += self.board[j, i + k]
                    tot += tmp_tot
                    # Controllo di quanti nella diagonale principale sono già
                    # Controllo altra diagonale
        # Mi serviva questo sleep solo per dei test, no capivo cosa non funzionava
        # time.sleep(60)
        return tot/8  # return moooolto provvisorio

    def fast_eval(self) -> int:
        """
        This function computes the number of possible 4 in a row that each player can
        still make and returns the difference
        """
        # Function that counts all possible sequences a player can make
        def count_open_sequences(player):
            count = 0
            # Horizontal
            for r in range(self.nrow):
                for c in range(self.ncol - 3):
                    window = self.board[r, c:c+4]
                    if is_valid_window(window, player):
                        count += 1
            # Vertical
            for r in range(self.nrow - 3):
                for c in range(self.ncol):
                    window = self.board[r:r+4, c]
                    if is_valid_window(window, player):
                        count += 1
            # Diagonal (top-left to bottom-right)
            for r in range(self.nrow - 3):
                for c in range(self.ncol - 3):
                    window = [self.board[r+i, c+i] for i in range(4)]
                    if is_valid_window(window, player):
                        count += 1
            # Diagonal (top-right to bottom-left)
            for r in range(self.nrow - 3):
                for c in range(3, self.ncol):
                    window = [self.board[r+i, c-i] for i in range(4)]
                    if is_valid_window(window, player):
                        count += 1
            return count

        # Function that checks if a player can make a sequence in the given window
        def is_valid_window(window, player):
            """
            Checks if a window contains only the player's pieces and empty spaces.
            """
            return all(cell == player or cell == 0 for cell in window)

        # Count potential sequences for each player
        player1_count = count_open_sequences(1)
        player2_count = count_open_sequences(2)

        # Return the difference
        return player1_count - player2_count

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
        player1_threats = find_threats(1)
        player2_threats = find_threats(2)

        # Filter threats
        player1_threats = filter_threats(player1_threats, player2_threats)
        player2_threats = filter_threats(player2_threats, player1_threats)

        # Calculate scores
        player1_score = score_threats(player1_threats)
        player2_score = score_threats(player2_threats)

        # Return the difference in threat scores
        return player1_score - player2_score

    def minimax(self, depth, debug=False) -> tuple[int, float]:
        """
        Minimax.
        """
        if self.has_ended == 1 or self.has_ended == -1:
            return (self.history[-1], float("inf")*self.has_ended)

        if self.legal_moves() is None:
            return (self.history[-1], 0)

        if depth == 0:
            return (self.history[-1], self.fast_eval() + self.threats_eval())

        curr_pl = self.curr_player()
        # Set best move as a random (the first) legal move, update later
        best_move = self.legal_moves()[0]
        # Maxplayer
        if curr_pl == MAXPLAYER:
            best_val = float("-inf")
            for move in self.legal_moves():
                self.make_move(move)

                _, new_val = self.minimax(depth - 1)
                if depth == DEBUG_DEPTH and debug:
                    print(f"{move}:{new_val} player:{
                          self.curr_player_name()} ")

                if new_val > best_val:
                    best_val = new_val
                    best_move = move

                self.undo_move()
        # Minplayer
        else:
            best_val = float('+inf')
            for move in self.legal_moves():
                self.make_move(move)

                _, new_val = self.minimax(depth - 1)
                if depth == DEBUG_DEPTH and debug:
                    print(f"{move}:{new_val} player:{self.curr_player_name()}")

                if new_val < best_val:
                    best_val = new_val
                    best_move = move

                self.undo_move()

        return (best_move, best_val)

    def alphabeta(self, depth, alpha=-100000, beta=100000) -> tuple[int, float]:
        """
        Minimax with alpha-beta pruning.
        """
        if self.has_ended == 1 or self.has_ended == -1:
            return (self.history[-1], float("inf")*self.has_ended)

        if self.legal_moves() is None:
            return (self.history[-1], 0)

        if depth == 0:
            return (self.history[-1], self.fast_eval() + self.threats_eval())

        curr_pl = self.curr_player()
        # Set best move as a random (the first) legal move, update later
        best_move = self.legal_moves()[0]

        if curr_pl == MAXPLAYER:
            best_val = float("-inf")
            for move in self.legal_moves():
                self.make_move(move)
                _, value = self.alphabeta(depth - 1, alpha, beta)
                self.undo_move()

                if value >= best_val:
                    best_val = value
                    best_move = move

                alpha = max(alpha, value)  # update lower bound
                if alpha > beta:
                    break
        else:
            best_val = float('+inf')
            for move in self.legal_moves():
                self.make_move(move)
                _, value = self.alphabeta(depth - 1, alpha, beta)
                self.undo_move()

                if value <= best_val:
                    best_val = value
                    best_move = move

                beta = min(beta, value)  # Update upper bound
                if beta < alpha:
                    break

        # print(f"{best_move}:{best_val} ")
        return (best_move, best_val)


if __name__ == "__main__":
    # b = Board()
    # b.make_move_sequence(
    #     [6, 3, 2, 3, 2, 3, 3, 3, 2, 3, 2],
    #     verbose=True)
    # print(b.has_ended)
    # print(b.legal_moves())
    # print(b.column_limits)
    # print(b)

    time_ditribiution = []

    test = Board()
    while test.has_ended == 0:
        # print(test)
        # if test.curr_player() == MINPLAYER:
        #     x = input("Waiting for your move: ")
        #     test.make_move(int(x))
        # else:
        start = time.time()
        move = test.alphabeta(DEBUG_DEPTH)[0]
        end = time.time()
        time_ditribiution.append(end-start)
        print("Played: ", move, "player", test.curr_player_name())
        test.make_move(move)
        print(test)

    print(test.history)
    print(test.has_ended)
    print(time_ditribiution)
