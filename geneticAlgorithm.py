import re
import random
import time
import matplotlib.pyplot as plt
# import numpy as np

CAPACITY = 0
WEIGHTS = []
VALUES = []

SizeOfPopulation = 0
NumberOfGeneration = 0
CrossoverProbability = 0
NumberOfcrossoverPoints = 0
MutationProbability = 0

NumberOfGenes = 4
NumberOfchild = 2


#################### ArgV ########################
def fetchArgvFromFile(argvFileName) :
    global CAPACITY, WEIGHTS, VALUES
    with open(argvFileName, 'r') as FILE :
        argv = []
        for line in FILE:
            value = line.split(':')[1]
            argv.append(value)
            
        CAPACITY = int(argv[0])
        WEIGHTS = map( int, re.findall(r'\d+', argv[1]) )
        VALUES = map( int, re.findall(r'\d+', argv[2]) )


def fetchArgvFromUser() :
    global SizeOfPopulation, NumberOfGeneration, CrossoverProbability, MutationProbability, NumberOfcrossoverPoints
    
    while True :
        SizeOfPopulation = int( input("Size of population: ") )
        if SizeOfPopulation < 2:
            print "Minimum size of population is 2"
        else :
            break 

    NumberOfGeneration = int(input("Number of generation: "))

    while True :
        CrossoverProbability = float( input("Crossover probability: ") )
        if CrossoverProbability > 1 or CrossoverProbability < 0 :
            print "Probability must be a number 0-1"
        else :
            break   

    while True :
        NumberOfcrossoverPoints = int( input("Crossover points: ") )
        if NumberOfcrossoverPoints > NumberOfGenes - 1:
            print "Too much crossover points"
        else :
            break

    while True :
        MutationProbability = float( input("Mutation probability: ") )
        if MutationProbability > 1 or MutationProbability < 0 :
            print "Probability bust be a number 0-1"
        else :
            break



###########################################################
def generatePopulation() :
    genes = []
    population = []

    #create genes list
    for i in range( len(WEIGHTS) ) :
        genes.append( (WEIGHTS[i], VALUES[i]) )

    #create individuals
    while len(population) < SizeOfPopulation :
        chromosome = []

        for i in range(NumberOfGenes) :
            while True :
                gene = genes[ random.randint(0, len(genes)-1) ]
                if gene not in chromosome: 
                    break
            chromosome.append(gene)

        population.append(chromosome)

    return population


def fintess(individual) :
    totalWeight = 0
    totalValue = 0

    for i in individual:
        totalWeight += i[0]
        totalValue += i[1]

    if totalWeight > CAPACITY :
        return 0
    else :
        return totalValue
        

def parentSelection(population) :
    fitnessList = []
    rouleteWheel = []
    parents = []

    #measure fitness
    for individual in population :
        fitnessList.append( fintess(individual) )

    #build roulette wheel
    for index, i in enumerate(fitnessList) :
        if i is not 0 :
            percent = round( (float(i) / sum(fitnessList)) * 100 )
            for j in range( int(percent) ) :
                rouleteWheel.append(index)

    if all( x == rouleteWheel[0] for x in rouleteWheel ) :
        return None

    #spin the wheel
    while len(parents) < 2 :
        parent = random.choice(rouleteWheel)
        if parent not in parents :
            parents.append(parent)

    return parents
    

def reproduction(parents) :
    parent1 = population[ parents[0] ]
    parent2 = population[ parents[1] ]
    children = []

    chromosomLength = len(parent1)
    crossoverPoints = []        

    #generate random crossing points
    while len(crossoverPoints) < NumberOfcrossoverPoints :
        crossoverPoint = random.choice( range(chromosomLength-1) )

        if crossoverPoint not in crossoverPoints :
            crossoverPoints.append(crossoverPoint)

    crossoverPoints.sort()
    crossoverPoints = [x+1 for x in crossoverPoints]
    crossoverPoints.insert(0, 0)
    crossoverPoints.append(len(parent1))

    #crossingover
    if CrossoverProbability > random.random() :
        for childNr in range(NumberOfchild) :
            child = []
            for index, i in enumerate(crossoverPoints) :
                if i != crossoverPoints[-1] :
                    if index % 2 != childNr % 2  :
                        child.extend( parent1[ i : crossoverPoints[index+1] ] )
                    else :
                        child.extend( parent2[ i : crossoverPoints[index+1] ] )

            if MutationProbability > random.random() :
                child = mutation(child)

            children.append(child)
        return children
    else :
        return None

def mutation(child) :
    #swap mutation
    x, y = random.sample(child, 2)
    child[ child.index(x) ] = y
    child[ child.index(y) ] = x
    return child


def survivalSelection(population, children) :
    survivors = population
    fitnessList = []

    for individual in population :
        fitnessList.append( fintess(individual) )    
    
    #sorted population according to fitness
    survivors = [ x for _,x in sorted(zip(fitnessList, population)) ]

    #remove least fitting, the fittest stay
    for i in range( len(children) ):
        if len(survivors) > 1 :
            survivors.pop(0)
        
    for index, i in enumerate(population) :
        if i not in survivors :
            population[index] = children.pop()
    
    return population


##########################################################
def init(argvFilename) :
    fetchArgvFromFile(argvFilename)
    fetchArgvFromUser()

    population = generatePopulation()

    return population

def evolution(population) :
    for i in range(NumberOfGeneration) :
        parents = parentSelection(population)
        if parents != None :
            children = reproduction(parents)
            if children != None :
                population = survivalSelection(population, children)
    
    fitnessList = []
    for individual in population :
        fitnessList.append( fintess(individual) )  

    return population[ fitnessList.index(max(fitnessList)) ]


if __name__ == '__main__':
    argvFilename = "argv"

    population = init(argvFilename)
    bestSolution = evolution(population)

    print "Best solution: ", bestSolution
    

##################### benchmark #####################################
else :
    testDataSets = [
#0
        [
            [10, 10, 0.9, 1, 0.1],
            [50, 10, 0.9, 1, 0.1],
            [100, 10, 0.9, 1, 0.1],
            [200, 10, 0.9, 1, 0.1]
        ],
#1
        [
            [10, 200, 0.9, 1, 0.1],
            [50, 200, 0.9, 1, 0.1],
            [100, 200, 0.9, 1, 0.1],
            [200, 200, 0.9, 1, 0.1]
        ],
#2
        [
            [100, 10, 0.9, 1, 0.1]
        ],
 #3 
        [
            [10, 100, 0.9, 1, 0.1]
        ],
#4   
        [
            [100, 100, 0.9, 1, 0.1]
        ],
#5
        [
            [100, 100, 0.9, 2, 0.1]
        ],
#6 
        [
            [100, 100, 0.9, 3, 0.1]
        ],
#7
        [
            [100, 100, 0.8, 1, 0.5]
        ],
 #8
        [
            [100, 100, 0.8, 2, 0.5]
        ],
#9
        [
            [100, 10, 0.9, 2, 0.1]
        ],
#10 
        [
            [100, 1000, 0.9, 2, 0.1]
        ]
    ]


    fetchArgvFromFile("argv")

    allWeights = []
    allValues = []
    fitnessList = []

    numberOfAttempts = 100
    numberOfSet = 10

    for attempt in range(numberOfAttempts) :
        for i in testDataSets[numberOfSet] :
            SizeOfPopulation = i[0]
            NumberOfGeneration = i[1]
            CrossoverProbability = i[2]
            NumberOfcrossoverPoints = i[3]
            MutationProbability = i[4]

            population = generatePopulation()
            bestSolution = evolution(population)

            total_weight = 0
            total_value = 0

            for i in bestSolution :
                total_weight += i[0]
                total_value += i[1]

            allWeights.append(total_weight)
            allValues.append(total_value)
            fitnessList.append( fintess(bestSolution) ) 
   
  
    plt.xlabel("WEIGHTS")
    plt.ylabel("VALUES")
    plt.xlim(right = CAPACITY + 1)
    plt.plot(allWeights, allValues, 'o')
    plt.suptitle("Fitness plot \nBest solutions: %d" % numberOfAttempts)
    # plt.show()
    plt.savefig('%d_solutions.png' % numberOfSet)
    plt.close()
        
    fitnessPercent = []
    for i in fitnessList :
        fitnessPercent.append( (float(i) / sum(fitnessList)) * 100 )
    plt.xlim(right=numberOfAttempts)
    plt.plot(fitnessPercent, '.-')
    plt.suptitle("Fitness plot \nNumber of attempts: %d" % numberOfAttempts)
    # plt.show()
    plt.savefig('%d_fitness.png' % numberOfSet)



##### executon time ##################

    executionTimeTestSets = [
        [
            [10, 10, 0.9, 1, 0.1]
        ],
        [
            [10, 200, 0.9, 1, 0.1]
        ],
        [
            [200, 200, 0.9, 1, 0.1]
        ],

        [
            [100, 100, 0.9, 2, 0.1]
        ],
        [
            [100, 100, 0.9, 3, 0.1]
        ],

        [
            [200, 200, 1, 3, 1]
        ],
        [
            [200, 1000, 1, 3, 1]
        ]
    ]

    for dataSet in executionTimeTestSets :
        for i in dataSet :
            SizeOfPopulation = i[0]
            NumberOfGeneration = i[1]
            CrossoverProbability = i[2]
            NumberOfcrossoverPoints = i[3]
            MutationProbability = i[4]
            
            start = time.time()

            population = generatePopulation()
            evolution(population)

            end = time.time()
            executionTime = end - start
        print "Test data: %s \nExecution time: %f" % (dataSet, executionTime)
        print ""
 