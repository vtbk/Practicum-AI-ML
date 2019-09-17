import copy
class State:
    def __repr__(self):
        return self.selected_string()

    def __init__(self, board, selected):
        self.board, self.selected = board, copy.deepcopy(selected)
    
    def selected_string(self):
        return ''.join(self.board[select[0]][select[1]] for select in self.selected) #Constructs the currently selected string by combining the letters of each selected cell

    def new_index(self, index, boundary, mutation): 
        if index + mutation > boundary:
            return 0
        elif index + mutation < 0:
            return boundary
        return index + mutation

    def ancestors(self):
        last, length = self.selected[-1], len(self.board)
        ancestors = []
        for direction in [[0, -1], [0, 1], [-1, 0], [1, 0]]:
            x = self.new_index(last[0], length - 1, direction[0])
            y = self.new_index(last[1], length - 1, direction[1])
            if [x, y] not in self.selected:
                updated_selected = copy.deepcopy(self.selected)
                updated_selected.append([x, y])
                ancestors.append(State(self.board, updated_selected))
        return ancestors

def generate_prefix_list(words):
    prefixes = []
    for word in words:
        for i in range(1, len(word)):
            prefixes.append(word[:i])
    return prefixes

def dfs(state, prefixes, words):
    if state.selected_string().lower() not in prefixes:
        return None
    elif state.selected_string().lower() in words:
        return [state]

    hits = []
    for child in state.ancestors():
        result = dfs(child, prefixes, words)
        if result != None:
            hits += result
    return hits

def solve(board, words):
    found = []
    prefixes = generate_prefix_list(words)
    for x in range(0, len(board)):
        for y in range (0, len(board[x])):
            state = State(board, [[x,y]]) #Create a state where this cell is the first one selected
            hits = dfs(state, prefixes, words) 
            if hits != None:
                found += hits
    return found

board = [['B', 'D', 'G', 'O'], ['O', 'E', 'N', 'M'], ['T', 'R', 'U', 'P'], ['I', 'X', 'Y', 'W']]
with open("words.txt", "r") as f:
    words = f.read().splitlines()
print solve(board, words)

