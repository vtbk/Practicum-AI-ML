import itertools

inhabitants = ('L', 'M', 'N', 'E', 'J')

constraints = []
constraints.append(lambda floors: floors[4] != 'L')
constraints.append(lambda floors: floors[0] != 'M')
constraints.append(lambda floors: floors[0] != 'N' and floors[4] != 'N')
constraints.append(lambda floors: floors.index('E') - floors.index('M') >= 1)
constraints.append(lambda floors: abs(floors.index('N') - floors.index('M')) > 1)
constraints.append(lambda floors: abs(floors.index('J') - floors.index('N')) > 1)

def verify_floors(floors):
    for constraint in constraints:
        if not constraint(floors):
            return False
    return True

def solve():
    for floors in list(itertools.permutations(inhabitants)):
        if verify_floors(floors): 
            return floors

print(solve())
