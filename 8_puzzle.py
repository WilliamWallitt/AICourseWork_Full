import math
import copy
import time
import numpy as np

# Heuristics - Manhatten and Euclidean
def calcuateManhattenHeuristics(state1, state2, grid_size):

    m = 0
    goal_state = []

    for i in range(len(state2)):
        for j in range(len(state2)):
            goal_state.append(state2[i][j])

    for i in range(0, grid_size, 1):
        for j in range(0, grid_size, 1):

            current_tile_value = state1[i][j]
            current_i, current_j = i, j

            indexOfGoalStatesTile = goal_state.index(current_tile_value)
            goal_i, goal_j = indexOfGoalStatesTile // grid_size, indexOfGoalStatesTile % grid_size

            if current_tile_value != 0:
                m += math.fabs(current_i - goal_i) + math.fabs(current_j - goal_j)

    return m


def calculateEucledianDistanceHeuristic(state1, state2, grid_size):

    euclid = 0

    for row in range(0, grid_size, 1):

        for col in range(0, grid_size, 1):

            current_val = state1[row][col]

            if current_val == 0:
                continue

            goal_coords = findTilePosition(state2, current_val)

            rowdiff = (row - goal_coords[0]) ** 2
            coldiff = (col - goal_coords[1]) ** 2

            euclid += math.sqrt(rowdiff + coldiff)

    return euclid


def findTilePosition(state, val, grid_size):

    for row in range(0, grid_size, 1):
        for col in range(0, grid_size, 1):
            if state[row][col] == val:
                return [row, col]


# function that calculates all possible moves of a specific state

def generatePossibleMoves(state, grid_size):

    possibleMoves = []
    row, column = 0, 0

    for r in range(0, grid_size, 1):
        for c in range(0, grid_size, 1):
            if state[r][c] == 0:
                row, column = r, c

    if row > 0:

        down_node = copy.deepcopy(state)
        down_node[row][column] = down_node[row - 1][column]
        down_node[row - 1][column] = 0
        possibleMoves.append((down_node, "up"))

    if column > 0:

        right_node = copy.deepcopy(state)
        right_node[row][column] = right_node[row][column - 1]
        right_node[row][column - 1] = 0
        possibleMoves.append((right_node, "left"))

    if row < (grid_size - 1):

        up_node = copy.deepcopy(state)
        up_node[row][column] = up_node[row + 1][column]
        up_node[row + 1][column] = 0
        possibleMoves.append((up_node, "down"))

    if column < (grid_size - 1):

        left_node = copy.deepcopy(state)
        left_node[row][column] = left_node[row][column + 1]
        left_node[row][column + 1] = 0
        possibleMoves.append((left_node, "right"))

    return possibleMoves


# each state/node has certain features, this class contains the attributes of each node
# it also has the main A* algorithm as a method of the class called solve function

class State:

    def __init__(self, inital_state=None):

        # constructor for each state
        self.state = inital_state
        # f = evaluation function
        # g = operating cost function
        # h = heuristic function
        self.h = 0
        self.g = 0
        self.f = 0
        self.parent = None
        self.action = None

    # A* Algorithm
    def solveProblem(self, initial, goal, heuristic_function, grid_size):

        timer = time.process_time()

        # keeping track of certain useful data
        generated_nodes_count = 0
        expanded_nodes_count = 0
        # closed = nodes that we have closed
        closed = []
        # open = what node we are going to visit next (this will be sorted by f value)
        open = []

        if initial.state == goal.state:

            printResults(initial, expanded_nodes_count, generated_nodes_count)

            return

        if heuristic_function == 1:

            initial.h = calcuateManhattenHeuristics(initial.state, goal.state, grid_size)

        elif heuristic_function == 2:

            initial.h = calculateEucledianDistanceHeuristic(initial.state, goal.state, grid_size)

        initial.f = initial.h + initial.g
        initial.parent = None
        initial.action = None

        open.append(initial)

        # check if initial state is solvable before we run the main search code
        if not isSolvable(initial.state):

            printMatrix(initial.state)
            print("No Solution Found for this configuration")

            return

        while open:

            current_node = open.pop(0)

            possibleNeighbors = generatePossibleMoves(current_node.state, grid_size)
            closed.append(current_node)
            expanded_nodes_count += 1

            for neighbor in possibleNeighbors:

                child_node = State()
                child_node.state = neighbor[0]
                child_node.action = neighbor[1]
                child_node.g = current_node.g + 1

                if heuristic_function == 1:

                    child_node.h = calcuateManhattenHeuristics(child_node.state, goal.state, grid_size)

                elif heuristic_function == 2:

                    child_node.h = calculateEucledianDistanceHeuristic(child_node.state, goal.state, grid_size)

                child_node.f = child_node.g + child_node.h
                child_node.parent = current_node

                generated_nodes_count += 1

                if child_node.state == goal.state:

                    print("------------------")
                    print("Solution Found!")
                    print("Time taken: " + str(round((time.process_time() - timer), 2)))
                    printResults(child_node, expanded_nodes_count, generated_nodes_count, grid_size)

                    return

                # this goes through all the open nodes and returns true if open false if not
                isExpanded = isInExpanded(closed, child_node)
                # if the node is not in the closed list
                if not isExpanded:

                    found = False
                    k = 0

                    for item in open:
                        if item.state == child_node.state:
                            found = True
                            if child_node.f < item.f:
                                item.f = child_node.f
                                open[k] = item
                                break
                        k += 1

                    if not found:
                        open.append(child_node)

                open = sorted(open, key=lambda x: x.f)

        print("No Solution Found!")
        return


# checks if the node is a member of the expanded node set
def isInExpanded(expanded, child_node):
    for node in expanded:
        if node.state == child_node.state:
            return True

    return False


# true means solvable false means not solvable
def isSolvable(state):

    inv_count = 0
    state_list = []

    for row in state:
        for tile in row:
            state_list.append(tile)

    for tile in state_list:

        if tile == 0:

            continue

        for item in state_list:

            if state_list.index(item) > state_list.index(tile) and item < tile and item != 0:

                inv_count += 1

    return inv_count % 2 == 0


# this backtracks up the nodes parents, and prints out each move (in reverse)
def pathToGoal(state):
    path = []

    for node in range(state.g):
        path.append(state)
        state = state.parent

    path.reverse()

    for i, p in enumerate(path):
        printMatrix(p.state)
        print("Step:" + str(i + 1) + " move -> " + p.action)
        print("--------------")

# this function prints out the results


def branchingFactor(nodes, depth):

    return nodes ** (1/depth)


def printResults(state, expanded_nodes, generated_nodes, grid_size):
    print("Generated Nodes Count:", generated_nodes)
    print("Expanded Nodes Count:", expanded_nodes)
    print("Node count:", state.g)
    print("Branching Factor:", round(branchingFactor(generated_nodes, state.g), 2))
    # print(h.heap())
    print("---------------------")
    pathToGoal(state)
    print(" -----------------------------------------")
    print("| Please enter 1 to run the program again |")
    print("| Please enter 2 to quit                  |")
    print(" -----------------------------------------")

    userinput = input("-> ")

    try:
        userinput = int(userinput)

    except ValueError:

        print("Error: Exiting now")
        return

    if int(userinput) == 1:
        startPuzzleSolver(grid_size)
        return
    elif int(userinput) == 2:
        return
    else:
        return


# this function prints out the state as a 3x3 matrix
def printMatrix(state):
    for s in state:
        print(str(s).strip('[]').replace(",", ''))


# the function starts the game
def startPuzzleSolver(grid_size):

    start_state_list, goal_state_list = startSolver(grid_size)

    if len(start_state_list) == 0 or len(goal_state_list) == 0:

        return

    print("Start State")
    printMatrix(start_state_list)
    print("End State")
    printMatrix(goal_state_list)

    intial_state = State(start_state_list)
    goal_state = State(goal_state_list)

    print(" ----------------------------------------------------")
    print("| Please enter 1 to use Manhattan Distance heuristic |")
    print("| Please enter 2 to use Euclidean Distance heuristic |")
    print("| Please enter 3 to quit                             |")
    print(" ----------------------------------------------------")

    userInput = input("-> ")

    if int(userInput) == 3:

        return

    if int(userInput) < 1 or int(userInput) > 2:
        print("----------------------")
        print("Error: Please enter a valid number")
        print("----------------------")

        startPuzzleSolver(grid_size)

        return

    intial_state.solveProblem(intial_state, goal_state, int(userInput), grid_size)


def startSolver(grid_size):

    start_state = []
    end_state = []

    print(" ------------------------------")
    print("Welcome to my 8-Puzzle Solver!")
    print(" ------------------------------")
    print("Please enter 1 to enter your own puzzle")
    print("Please enter 2 to use the test puzzle")
    print("Please enter 3 to quit")

    userinput = input("-> ")

    try:

        int(userinput)

        if int(userinput) == 3:

            return [], []

    except ValueError:

        print(" ------------------------------")
        print("Please enter either 1 or 2")
        print(" ------------------------------")

        startPuzzleSolver(grid_size)

        return [], []

    if int(userinput) == 1:

        print("Please enter space separated integers for your starting/end state")
        print("For example: '1 2 3 4 5 6 7 9 0'")

        start = input("Starting State -> ").split(" ")
        end = input("End State -> ").split(" ")

        try:

            start = [int(num) for num in start]
            end = [int(num) for num in end]

        except ValueError:

            print(" ----------------------------------")
            print("Error: Please enter a valid start and end matrix!")
            print(" ----------------------------------")

            startPuzzleSolver(grid_size)

        # start = [int(num) for num in start]
        # end = [int(num) for num in end]

        start = np.reshape(start, (grid_size, grid_size)).tolist()
        end = np.reshape(end, (grid_size, grid_size)).tolist()

        start_state = start
        end_state = end

        if int(userinput) == 1 and checkUserMatrixIsValid(start_state, end_state, grid_size):

            return start_state, end_state

        else:

            print(" ----------------------------------")
            print("Error: Please enter a valid start and end matrix!")
            print(" ----------------------------------")

            startPuzzleSolver(grid_size)

            return

    elif int(userinput) == 2:

        if grid_size == 4:

            start_state = [[1, 2, 4, 0], [5, 7, 3, 8], [9, 6, 10, 12], [13, 14, 11, 15]]
            end_state = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]
        else:
            start_state = [[7, 2, 4], [5, 0, 6], [8, 3, 1]]
            end_state = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        return start_state, end_state

    else:

        print(" ----------------------------------")
        print("Error: Please enter a valid number!")
        print(" ----------------------------------")

        startPuzzleSolver(grid_size)

        return


def checkUserMatrixIsValid(start_matrix, end_matrix, grid_size):

    num = [x for x in range((grid_size ** 2))]
    start_test = []
    end_test = []

    for row in start_matrix:
        for item in row:
            start_test.append(item)

    for row in end_matrix:
        for item in row:
            end_test.append(item)

    if num == sorted(start_test) and num == sorted(end_test):

        return True

    else:

        return False


def main():

    print("Please enter your N-puzzle grid size (number of tiles per row!)")
    grid_size = input("->")
    startPuzzleSolver(int(grid_size))

main()