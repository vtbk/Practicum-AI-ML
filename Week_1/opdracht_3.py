from heapq import heappush
from heapq import heappop
from collections import namedtuple
from copy import deepcopy
class State:
    def __eq__(self, other):
        if isinstance(other, State ) and self.board == other.board:
            return True
        return False

    def __repr__(self):
        repr = '\n'
        for row in range(0, self.length):
            for col in range(0, self.length):
                repr += str(self.board[col][row])
            repr += "\n"
        return repr

    def __lt__(self, other): return True

    def __init__(self, board, previous = None, depth = 0):
        self.board, self.previous, self.depth = board, previous, depth
        self.length = len(board)
        self.coordinate = namedtuple('Coordinate', 'x y')

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
        for x, row in enumerate(self.board):
            for y, col in enumerate(row):
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
        visited = []
        heappush(frontier, (0, current_state))
        while len(frontier) > 0:
            current = heappop(frontier)[1]
            if current == desired_state:
                return current
            visited.append(current)
            for ancestor in current.ancestors():
                if ancestor not in visited:
                    heappush(frontier, (self.heuristic(ancestor, desired_state) + ancestor.depth, ancestor))


def heuristic(current_state, desired_state):
    total = 0
    for x, row in enumerate(current_state.board):
        for y, val in enumerate(row):
            desired_location = desired_state.coords_of(val)
            total += abs(x - desired_location.x) + abs(y - desired_location.y)
    return total


solver = Solver(heuristic)
board, desired = [[1, 4, 7], [2, 0, 8], [3, 5, 6]], [[1, 4, 7], [2, 5, 8], [3, 6, 0]] #Board is a multidimensional array

r = solver.solve(State(board), State(desired))

print("Winning: ")
while r != None:
    print(r)
    r = r.previous
#maybe set parent on state so that you can get the actual path instead of just the amout of steps