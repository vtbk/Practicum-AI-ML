from heapq import heappush
from heapq import heappop
from collections import namedtuple
from copy import deepcopy
import cProfile
class State:
    coordinate = namedtuple('Coordinate', 'x y')

    def __hash__(self):
        return hash(str(self.board))

    def __eq__(self, other):
        if isinstance(other, State) and self.board == other.board:
            return True
        return False

    def __repr__(self): #Yuck! TODO: Change board representation from board[x][y] to board[y][x]; will turn this into a simple str join.
        temp = ''
        for x in range(0, self.length):
            for y in range(0, self.length):
                temp += str(self.board[y][x]) + " "
            temp += '\n'
        return temp

    def __lt__(self, other): return True

    def __init__(self, board, previous = None, depth = 0):
        self.board, self.previous, self.depth = board, previous, depth
        self.length = len(board)

    def around(self, coordinate):
        coords = []
        if coordinate.x > 0:
            coords.append(self.coordinate(coordinate.x - 1, coordinate.y))
        if coordinate.x < self.length - 1:
            coords.append(self.coordinate(coordinate.x + 1, coordinate.y))
        if coordinate.y > 0:
            coords.append(self.coordinate(coordinate.x, coordinate.y - 1))
        if coordinate.y < self.length - 1:
            coords.append(self.coordinate(coordinate.x, coordinate.y + 1))
        return coords
    
    def coords_of(self, value):
        for x in range(0, self.length):
            for y in range(0, self.length):
                if self.board[x][y] == value:
                    return self.coordinate(x, y)

    def ancestors(self):
        ancestors = []
        for coord in self.around(self.coords_of(0)):
            new_state = State(deepcopy(self.board), self, self.depth + 1)
            new_state.move(coord)
            ancestors.append(new_state)
        return ancestors
        
    def move(self, new_spot):
        empty = self.coords_of(0)
        self.board[empty.x][empty.y] = self.board[new_spot.x][new_spot.y]
        self.board[new_spot.x][new_spot.y] = 0

class Solver:
    def __init__(self, heuristic):
        self.heuristic = heuristic
        
    def solve(self, current_state, desired_state):
        frontier = []
        visited = set()
        heappush(frontier, (0, current_state))

        while len(frontier) > 0:
            current = heappop(frontier)[1]
            if current == desired_state:
                return current
            visited.add(current)
            for ancestor in current.ancestors():
                if ancestor not in visited:
                    heappush(frontier, (self.heuristic(ancestor, desired_state) + ancestor.depth, ancestor))


#---------------------#
#Solver requires injection of heuristic function
def heuristic(current_state, desired_state):
    total = 0
    for x in range(0, current_state.length):
        for y in range(0, current_state.length):
            val = current_state.board[x][y]
            desired_location = desired_state.coords_of(val)
            total += abs(x - desired_location.x) + abs(y - desired_location.y)
    return total

#Useless heuristic to demonstrate inefficiency of Dijkstra
def mocked_heuristic(current_state, desired_state):
    return 0


board = [[8, 2, 3], [6, 5, 0], [7, 4, 1]]
desired = [[1, 4, 7], [2, 5, 8], [3, 6, 0]]

solver = Solver(heuristic)
solution = solver.solve(State(board), State(desired))
print("Solution found at depth: " + str(solution.depth))

state = solution
while state.previous != None:
    print("Move #" + str(state.depth))
    print(state)
    state = state.previous
