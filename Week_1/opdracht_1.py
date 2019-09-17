import copy

class Node:
    def out(self):
        print ', '.join(self.state[0]) + " \~~~~~~~~~~~/ " + ', '.join(self.state[1])

    def __eq__(self, other):
        if self.state == other.state:
            return True
        return False

    def __init__(self, parent):
        self.state = copy.deepcopy(parent.state) if parent != None else (set(['C', 'G', 'W', 'F']), set([]))
        self.parent = parent

    def farmerLocation(self):
        return 0 if 'F' in self.state[0] else 1

    #Find possible next states
    def children(self):
        children = []
        farmerBank = self.farmerLocation()
        oppositeBank = 1 if farmerBank == 0 else 0

        for actor in self.state[farmerBank]: #A non-farmer actor can only cross when the farmer is one the same side as them
            child = Node(self)
            #Farmer always moves to the opposite side, no matter whether he brings an actor with him or not
            child.state[farmerBank].remove('F')
            child.state[oppositeBank].add('F')
            if actor != 'F':
                child.state[farmerBank].remove(actor)
                child.state[oppositeBank].add(actor)
            children.append(child)
        return children
    
def finished(node):
    if node.state[0] == set(): #If the left bank is empty the puzzle is solved
        return True
    return False

def valid(node):
    for side in node.state:
        if set(['C', 'G']) == side or set(['G', 'W']) == side: #Check if any one side contains an invalid combination 
            return False
    return True

def dfs(node, path, depth):
    path.append(node)

    if finished(node):
        return [path]

    paths = []
    for child in node.children():
        if child not in path:
            if valid(child):
                childPaths = dfs(child, copy.deepcopy(path), depth - 1)
                paths += childPaths
    return paths

n = Node(None)
paths = dfs(n, [], 10)
for path in paths:
    print "\nPath: "
    for node in path:
        node.out()


#Notes:
#Infinite recursion would require setting a max depth or checking parents when returning children (weed out duplicates), if not comparing actual state, but only the node itself. 
#Problem: not really a node anymore like this; nodes are compared on state as opposed to state + parent
#Currently not finding the same node anyways, only the same state in different nodes. 
#Comparing actual nodes (parent and all) sounds like it's only relevant when graph is already made beforehand - you won't create matches recursively

#Cache not as nice as it could be right now; checks whether pattern occured higher up in path, but not on another branch in the tree


        
    

