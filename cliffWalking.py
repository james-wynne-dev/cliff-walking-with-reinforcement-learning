import numpy as np
import random
import matplotlib.pyplot as plt

MAP_WIDTH = 12
MAP_HEIGHT = 4
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 1
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
TERMINAL_STATE = [11, 0]

class Square:
    def __init__(self, xCoord, yCoord):
        self.qValues = np.zeros(4)
        self.theCliff = False
        self.reward = -1
        self.xCoord = xCoord
        self.yCoord = yCoord

    def chooseAction(self, epsilon=0.1):
        if random.random() <= epsilon:
            return random.randint(0,3)
        else:
            return np.random.choice([i for i in range(4) if self.qValues[i] == np.amax(self.qValues)])


class Map:
    def __init__(self):
        self.grid = [[Square(j, i) for i in range(MAP_HEIGHT)] for j in range(MAP_WIDTH)]
        for i in range(1, MAP_WIDTH - 1):
            self.grid[i][0].theCliff = True
        for i in range(1,11):
            self.grid[i][0].reward = -100
        self.currentSquare = [0, 0]

    def move(self, action):
        if action == UP:
            self.currentSquare[1] = min(MAP_HEIGHT - 1, self.currentSquare[1] + 1)
        if action == RIGHT:
            self.currentSquare[0] = min(MAP_WIDTH - 1, self.currentSquare[0] + 1)
        if action == DOWN:
            self.currentSquare[1] = max(0, self.currentSquare[1] - 1)
        if action == LEFT:
            self.currentSquare[0] = max(0, self.currentSquare[0] - 1)

    def getCurrentSquare(self):
        return self.grid[self.currentSquare[0]][self.currentSquare[1]]


def sarsa(numEpisodes, epsilon=0.1):
    map = Map()
    rewards = np.zeros(numEpisodes)
    for i in range(numEpisodes):
        map.currentSquare = [0, 0]
        totalReward = 0
        state = map.getCurrentSquare()
        action = state.chooseAction()
        moves = 0
        while (map.currentSquare != TERMINAL_STATE):
            moves += 1
            map.move(action)
            statePrime = map.getCurrentSquare()
            actionPrime = statePrime.chooseAction(epsilon)
            # calculateSarsaQ
            actionValue = state.qValues[action]
            actionValuePrime = statePrime.qValues[actionPrime]
            reward = statePrime.reward
            newQ = actionValue + (ALPHA *(reward + (GAMMA * actionValuePrime) - actionValue))
            state.qValues[action] = newQ
            totalReward += reward
            state = statePrime
            action = actionPrime
            if state.theCliff:
                break
        # limit lowest value to -100
        rewards[i] = max(totalReward, -100)
    return rewards, map


def qLearning(numEpisodes, epsilon=0.1):
    map = Map()
    rewards = np.zeros(numEpisodes)
    for i in range(numEpisodes):
        map.currentSquare = [0, 0]
        totalReward = 0
        moves = 0
        while (map.currentSquare != TERMINAL_STATE):
            moves += 1
            state = map.getCurrentSquare()
            action = state.chooseAction(epsilon)
            map.move(action)
            # calculate newQ
            q_SA = state.qValues[action]
            reward = map.getCurrentSquare().reward
            bestQ_sPrime = np.amax(map.getCurrentSquare().qValues)
            newQ = q_SA + (ALPHA * (reward + ((GAMMA * bestQ_sPrime) - q_SA)))
            state.qValues[action] = newQ
            totalReward += reward
            if state.theCliff:
                break
        rewards[i] = max(totalReward, -100)
    return rewards, map


def testAlgorithm(numRuns, numEpisodes, algorithm, epsilon=0.1):
    rewards = np.zeros((numRuns, numEpisodes))
    maps = np.empty([numRuns], map)
    qValues = np.zeros((MAP_WIDTH, MAP_HEIGHT, 4))
    for run in range(numRuns):
        rewards[run], maps[run] = algorithm(numEpisodes, epsilon)
    averagedRuns = np.mean(rewards, 0)
    for i in range(numRuns):
        for j in range(MAP_WIDTH):
            for k in range(MAP_HEIGHT):
                qValues[j][k] += maps[i].grid[j][k].qValues
    qValues /= numRuns
    movingAverage = np.zeros(numEpisodes)
    for i in range(numEpisodes):
        movingAverage[i] = np.mean(averagedRuns[max(i - 10, 0):i + 1])
    return movingAverage, qValues


def printBestActions(qValues):
    for i in range(3,-1,-1):
        for j in range(12):
            bestAction = np.argmax(qValues[j][i])
            if bestAction == UP:
                print("U", end="")
            elif bestAction == RIGHT:
                print("R", end="")
            elif bestAction == DOWN:
                print("D", end="")
            elif bestAction == LEFT:
                print("L", end="")
        print("")

def testGraph(numRuns, numEpisodes, algorithm, epsilon, col, lab):
    rewards, qValues = testAlgorithm(numRuns, numEpisodes, algorithm, epsilon)
    plt.plot(rewards, color=col, label=lab)

# ------------- Sarsa at different values of epsilon ----------------
plt.figure(1)
testGraph(10, 500, sarsa, 0.01, 'red', "sarsa ε=0.01")
testGraph(10, 500, sarsa, 0.1, 'blue', "sarsa ε=0.1")
testGraph(10, 500, sarsa, 0.3, 'green', "sarsa ε=0.3")
testGraph(10, 500, sarsa, 0.5, 'yellow', "sarsa ε=0.5")
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend()

# ------------- Q-Learning at different values of epsilon ------------------
plt.figure(2)
testGraph(10, 500, qLearning, 0.01, 'red', "Qlearn ε=0.01")
testGraph(10, 500, qLearning, 0.1, 'blue', "Qlearn ε=0.1")
testGraph(10, 500, qLearning, 0.3, 'green', "Qlearn ε=0.3")
testGraph(10, 500, qLearning, 0.5, 'yellow', "Qlearn ε=0.3")
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend()

# ------------- Sarsa Vs Q Learning at epsilon 0.01
plt.figure(3)
testGraph(10, 500, qLearning, 0.01, 'red', "Qlearn ε=0.01")
testGraph(10, 500, sarsa, 0.01, 'blue', "sarsa ε=0.01")
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend()

# ------------- Sarsa Vs Q Learning at epsilon 0.1 ----------------------
plt.figure(4)
testGraph(10, 500, qLearning, 0.1, 'red', "Qlearn ε=0.1")
testGraph(10, 500, sarsa, 0.1, 'blue', "sarsa ε=0.1")
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.legend()

plt.show()

# ------------- Best actions for Sarsa and Q-Learning ---------------------
rewardsSarsa, qValuesSarsa = testAlgorithm(10, 500, sarsa, 0.1)
rewardsQLearning, qValuesQLearning = testAlgorithm(10, 500, qLearning, 0.1)
print("Best Actions - Sarsa")
printBestActions(qValuesSarsa)
print("Best Actions - Q-Learning")
printBestActions(qValuesQLearning)
