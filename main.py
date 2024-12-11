import numpy as np
import time

MINPLAYER: int = -1
MAXPLAYER: int = 1
EMPTY: int = 0

#Se self.has_ended è uguale a 2 allora è patta


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
        if self.has_ended != 0:
            return None
        else:
            return np.where(self.column_limits < self.nrow)[0]

    def make_move(self, move: int) -> None:
        """
        Updates the current board rappresentation given the move: the column in which 
        to drop the piece. This function incrementally checks if the given move ends the game
        connecting four or more.
        """
        if self.has_ended != 0:
            raise Exception("Game already ended")

        if move not in self.legal_moves():
            raise Exception("Illegal move :(")

        curr_player = self.curr_player()
        row, col = self.column_limits[move], move
        self.board[row, col] = curr_player
        self.column_limits[col] += 1
        self.history.append(move)
        self.turn += 1

        connected_points = 0
        #Maybe we can check only the bottom for, because when we put a tile we put on the top so everthing over should be 0
        for crow in range(row - 3, row + 4):
            if crow < self.nrow and crow >= 0:
                if self.board[crow, col] == curr_player:
                    connected_points += 1
                    if connected_points >= 4:
                        self.has_ended = curr_player
                        return
                else:
                    connected_points = 0

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

        connected_points = 0
        for d in range(0, 4):
            if col + d < self.ncol and row - d >= 0:
                if self.board[row - d, col + d] == curr_player:
                    connected_points += 1
                    if connected_points >= 4:
                        self.has_ended = curr_player
                        return
                else:
                    connected_points = 0

        connected_points = 0
        for d in range(0, 4):
            if row + d < self.nrow and col - d >= 0:
                if self.board[row + d, col - d] == curr_player:
                    connected_points += 1
                    if connected_points >= 4:
                        self.has_ended = curr_player
                        return
                else:
                    connected_points = 0
        
        if self.legal_moves == None:
            self.has_ended = 2

        # Rendere anche la generazione delle mosse di forza quattro incementale
        # ogni volta prendo il max() della sequenza di vicini più lunga delle teste
        # che "faccio crescere" a quel punto mi basta controllare se il max(...) locale
        # arriva a 4.

        # Probabilmente ha senso farlo solo in versioni generalizzate del gioco in cui ho
        # sequenze arbitrarie da controllare

    def make_move_sequence(self, move_list: list[int], verbose: bool = False) -> None:
        """
        Plays a sequence of moves on the board
        """
        for move in move_list:
            if move in self.legal_moves():
                if verbose:
                    print(self)
                self.make_move(move)
                if self.has_ended != 0: break
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
        self.has_ended = EMPTY
    
    def eval(self) -> int:
        #Una posizione è forte quando:
        #calcolo il numero di elementi in riga in base alle caselle vuote che ha vicino
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
        #Probabilmente farò in parte controllo verticale in quanto molto più veloce
        for i in range(self.ncol - 1):
            #Dove controllare
            if self.column_limits[i + 1] > self.column_limits[i]:
                #Ciclo per ogni casella da controllare
                for j in range(self.column_limits[i], self.column_limits[i + 1] + 1):
                    tmp = 0
                    tmp_tot = 0
                    #Controllo di quanti orizzontali già in linea
                    for k in range(-3, 4):
                        if k == 0:  continue
                        if i + k >= 0 and i + k < self.ncol:
                            if self.board[j, i + k] == EMPTY:
                                #Se il valore assoluto di tmp è maggiore del valore assoluto di tot, allora scambia
                                #Forse questo posso modificare anche in base al turno del giocatore
                                if abs(tmp) > abs(tmp_tot):
                                    #Probabilmente devo controllare a chi tocca e devo anche pensare al numero di mosse che devo fare per arrivare fin là, ma per ora va bene così
                                    tmp_tot = tmp
                                tmp = 0
                            else:
                                tmp += self.board[j, i + k]
                    tot += tmp_tot
                    #Controllo di quanti nella diagonale principale sono già
                    #Controllo altra diagonale
        #Mi serviva questo sleep solo per dei test, no capivo cosa non funzionava
        #time.sleep(60)
        return tot/8    #return moooolto provvisorio

        pass


    def minimax(self, depth) -> tuple[int, float]:
        #Penso che farò un altro caso in cui depth <= 0, dove calcolerò un euristica ma per ora
        if self.has_ended == 1 or self.has_ended == -1 or depth <= 0:
            x = self.eval()
            return self.history[-1], self.has_ended
        if self.legal_moves() is None:
            return self.history[-1], 0
        curr_pl = self.curr_player()
        best = self.legal_moves()[0]
        if curr_pl == MAXPLAYER:
            a = float("-inf")
            for move in self.legal_moves():
                self.make_move(move)
                mossa, value = self.minimax(depth - 1)
                if value > a:
                    a = value
                    best = mossa
                self.undo_move()
        else:
            a = float('+inf')
            for move in self.legal_moves():
                self.make_move(move)
                mossa, value = self.minimax(depth - 1)
                if value < a:
                    a = value
                    best = mossa
                self.undo_move()
        return best, a

    def alphabeta(self) -> tuple[int, float]:
        pass


if __name__ == "__main__":
    b = Board()
    b.make_move_sequence(
        [1, 2, 0, 1, 5, 2, 1, 5, 3, 1, 4, 5, 6, 1, 3, 2, 4, 5, 6],
        verbose=True)
    print(b.has_ended)
    print(b.legal_moves())
    print(b.column_limits)
    print(b)

    prova = Board()
    while prova.has_ended == 0:
        print(prova)
        if prova.curr_player() == MINPLAYER:
            x = input("Waiting for your move: ")
            prova.make_move(int(x))
        else:
            prova.make_move(prova.minimax(5)[0])
    print(prova)
    print(prova.has_ended)