neighbours = dict()
neighbours[0] = [3]
neighbours[1] = [2]
neighbours[2] = [1, 3, 4]
neighbours[3] = [0, 2, 5]
neighbours[4] = [2, 5]
neighbours[5] = [3, 4, 6, 7]
neighbours[6] = [5]
neighbours[7] = [5]

ACE, JACK, KING, QUEEN, EMPTY = 'A', 'J', 'K', 'Q', ' '

def is_valid_board(board):
    for index, card in enumerate(board):
        neighbour_cards = [board[neighbour] for neighbour in neighbours[index]]
        if card == EMPTY:
            continue

        if card in neighbour_cards:
            return False

        if card == KING:
            if QUEEN in board and QUEEN not in neighbour_cards:
                return False
        
        if card == ACE:
            if KING in board and KING not in neighbour_cards:
                return False
            if QUEEN in neighbour_cards:
                return False

        if card == QUEEN:
            if JACK in board and JACK not in neighbour_cards:
                return False
    return True 

def is_finished_board(board):
    return EMPTY not in board

def get_empty_tiles(board):
    return [index for index, tile in enumerate(board) if tile == EMPTY]

def search(board, cards):
    if not is_valid_board(board):
        return False
    
    if is_finished_board(board):  
        return board
    
    for tile in get_empty_tiles(board):
        for card in cards:
            board[tile] = card
            cards.remove(card)
            result = search(board, cards)
            if result is False:
                board[tile] = EMPTY
                cards.append(card)
            else:
                return result
    return False

def tests():
    print(is_valid_board(['J', 'A', ' ', ' ', ' ', ' ', ' ', 'J']))
    print(not is_valid_board(['K', 'J', 'J', 'Q', 'Q', 'K', 'A', 'A']))
    print(is_valid_board(['K', 'Q', 'J', 'Q', 'A', 'K', 'J', 'A']))

board = [EMPTY for tile in range(8)]
print(search(board, [JACK, JACK, KING, KING, QUEEN, QUEEN, ACE, ACE]))

