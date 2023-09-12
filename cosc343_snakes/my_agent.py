__author__ = "<Max van der Sluis>"
__organization__ = "COSC343/AIML402, University of Otago"
__email__ = "<vanma236@student.otago.ac.nz>"

import numpy as np

agentName = "<my_agent>"
perceptFieldOfVision = 3  # Choose either 3,5,7 or 9
perceptFrames = 1  # Choose either 1,2,3 or 4
trainingSchedule = [("self", 25), ("random", 25)]

no_bias = perceptFieldOfVision
# This is the class for your snake/agent
class Snake:

    def __init__(self, nPercepts, actions):
        # You should initialise self.chromosome member variable here (whatever you choose it
        # to be - a list/vector/matrix of numbers - and initialise it with some random
        # values)
        self.chromosome = np.random.randint(low=-9, high=9, size=3*(perceptFieldOfVision**2)+no_bias)
        self.nPercepts = nPercepts
        self.actions = actions

    def AgentFunction(self, percepts):
        current_frame_percepts = percepts[0]
        count = 0
        a1 = 0
        a2 = 0
        a3 = 0
        # Calculating action weights
        for i in range(perceptFieldOfVision):
            for x in range(perceptFieldOfVision):
                a1 += current_frame_percepts[i-1][x-1] * self.chromosome[count]
                count += 1
        for i in range(perceptFieldOfVision):
            for x in range(perceptFieldOfVision):
                a2 += current_frame_percepts[i-1][x-1] * self.chromosome[count]
                count += 1
        for i in range(perceptFieldOfVision):
            for x in range(perceptFieldOfVision):
                a3 += current_frame_percepts[i-1][x-1] * self.chromosome[count]
                count += 1
        # Comparing weights and choosing action
        if a1 > a2 and a1 > a3:
            index = 0
        elif a2 > a1 and a2 > a3:
            index = 1
        else:
            index = 2
        return self.actions[index]


def evalFitness(population):
    N = len(population)

    # Fitness initialiser for all agents
    fitness = np.zeros((N))

    # This loop iterates over your agents in the old population - the purpose of this boiler plate
    # code is to demonstrate how to fetch information from the old_population in order
    # to score fitness of each agent
    for n, snake in enumerate(population):
        # snake is an instance of Snake class that you implemented above, therefore you can access any attributes
        # (such as `self.chromosome').  Additionally, the object has the following attributes provided by the
        # game engine:
        #
        # snake.size - list of snake sizes over the game turns
        # .
        # .
        # .
        maxSize = np.max(snake.sizes)
        turnsAlive = np.sum(snake.sizes > 0)
        maxTurns = len(snake.sizes)

        # This fitness functions considers snake size plus the fraction of turns the snake
        # lasted for.  It should be a reasonable fitness function, though you're free
        # to augment it with information from other stats as well
        fitness[n] = maxSize + turnsAlive / maxTurns

    return fitness


def newGeneration(old_population):
    # This function should return a tuple consisting of:
    # - a list of the new_population of snakes that is of the same length as the old_population,
    # - the average fitness of the old population

    N = len(old_population)

    nPercepts = old_population[0].nPercepts
    actions = old_population[0].actions

    fitness = evalFitness(old_population)

    # At this point you should sort the old_population snakes according to fitness, setting it up for parent
    # selection.
    p = []
    i = 0
    while i < len(fitness):
        p.append(fitness[i]/sum(fitness))
        i += 1
    # Create new population list...
    new_population = list()
    # Keeping the fittest 2 snakes for elitism
    high_indexes = np.argsort(p)[::-1][:2]
    new_population.append(old_population[high_indexes[0]])
    new_population.append(old_population[high_indexes[1]])
    for n in range(N-2):
        # Here you should modify the new snakes chromosome by selecting two parents (based on their
        # fitness) and crossing their chromosome to overwrite new_snake.chromosome
        # Selecting 2 parents with weighted random choice based on fitness
        p1, p2 = np.random.choice(N, size=2, replace=False, p=p)
        p3 = old_population[p1]
        p4 = old_population[p2]
        index = np.random.randint(30)
        # Create a new child snake
        c = Snake(nPercepts, actions)
        # Doing the crossover of parents into child
        c.chromosome = np.copy(p3.chromosome)
        c.chromosome[index:] = p4.chromosome[index:]
        # There is a 10% chance of a 10% of chromosome mutation
        mutate = np.random.randint(1, 10)
        if mutate == 5:
            for i in range(perceptFieldOfVision):
                c.chromosome[np.random.randint(0, 29)] = np.random.randint(-9, 9)
        # Add the new snake to the new population
        new_population.append(c)
    # At the end you need to compute the average fitness and return it along with your new population
    avg_fitness = np.mean(fitness)
    return new_population, avg_fitness

