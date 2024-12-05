import numpy as np

def differentialEvolutionEpoch(pop: list, cross_rate: float, scale_factor: float) -> list:
    '''Implements DE/best/1/z'''
    fitness = list()
    
    # Calculates the fitness of each network, depending on problem type
    for i in range(len(pop)):
        fitness.append(np.mean(pop[i].get_fitness()))

    # Selects the best network (x(t))
    best_index = fitness.index(max(fitness))

    # Selects the donor networks, making sure they are distinct
    first_index = np.random.randint(0, len(pop))
    while first_index == best_index:
        first_index = np.random.randint(0, len(pop))
    second_index = np.random.randint(0, len(pop))
    while second_index == best_index or second_index == first_index:
        second_index = np.random.randint(0, len(pop))

    # Calculates the mutated vector
    x_t = pop[best_index].get_chromosome()
    x_2 = pop[first_index].get_chromosome()
    x_3 = pop[second_index].get_chromosome()

    u_t = x_t + scale_factor*(x_2 - x_3)


    # Performs Binomial crossover
    for i in pop:
        parent = i.get_chromosome()
        which_parent = np.random.random(len(parent))
        offspring = np.where(which_parent <= cross_rate, u_t, parent)
        test = i
        test.set_weights(offspring)
        # If offsprings fitness is higher, replace original
        if np.mean(test.get_fitness()) >= np.mean(i.get_fitness()):
            i = test

    # Returns next generation
    return pop

def differentialEvolution(pop: list, cross_rate: float, scale_factor: float):
    fitness = list()
    
    # Calculates the fitness of each network, depending on problem type
    for i in range(len(pop)):
        fitness.append(np.mean(pop[i].get_fitness()))
    
    # Finds the best
    best = max(fitness)
    fitness.clear()

    # Runs through an epoch, finds the best
    offspring = differentialEvolutionEpoch(pop, cross_rate, scale_factor)
    for i in range(len(offspring)):
        fitness.append(np.mean(offspring[i].get_fitness()))
    best_off = max(fitness)

    # Continues until the two have converged
    counter = 0
    while (np.abs(best - best_off) / best_off <= 0.05):
        if counter == 100: 
            break
        counter+=1
        
        best = best_off
        fitness.clear()

        offspring = differentialEvolutionEpoch(pop, cross_rate, scale_factor)
        for i in range(len(offspring)):
            fitness.append(np.mean(offspring[i].get_fitness()))
        best_off = max(fitness)

    # Returns the best offspring and its fitness score
    index = fitness.index(best_off)
    return offspring[index], best_off