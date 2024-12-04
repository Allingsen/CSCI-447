import numpy as np

def tournament_sel(pop: int, networks: list, fitness: list) -> list:
    '''Performs Tournament selection'''
    new_pop = list()
    # Picks two random networks and compares fitness, selects the best
    for _ in range(pop):
        champ = np.random.randint(0, len(networks))
        chal = np.random.randint(0, len(networks))
        winner = max(fitness[champ], fitness[chal])
        # Adds winner to new population
        if winner == fitness[champ]:
            new_pop.append(networks[champ])
        else:
            new_pop.append(networks[chal])
    
    return new_pop

def geneticAlgEpoch(pop: list, cross_rate: float, mut_rate: float) -> list:
    '''Performs weight tuning using a Genetic Algorithm'''
    fitness = list()
    
    # Calculates the fitness of each network, depending on problem type
    for i in range(len(pop)):
        fitness.append(np.mean(pop[i].get_fitness()))

    # Creates a new population
    population = tournament_sel(len(pop), pop, fitness)
    updated_chrom_pop = list()

    # Performs Uniform crossover
    for i in range(0, len(pop), 2):
        # Selects parents (Chromosomes)
        parent_1 = population[i].get_chromosome()
        parent_2 = population[i+1].get_chromosome()
        if np.random.rand() <= cross_rate:
            # Each value is chosen randomly from the parents, each
            # having equal probalbility 
            which_parent_1 = np.random.randint(0,2, size=len(parent_1))
            which_parent_2 = np.random.randint(0,2, size=len(parent_2))
            updated_chrom_pop.append(np.where(which_parent_1 == 0, parent_1, parent_2))
            updated_chrom_pop.append(np.where(which_parent_2 == 0, parent_1, parent_2))
        else:
            # Else, just add the parents
            updated_chrom_pop.append(parent_1)
            updated_chrom_pop.append(parent_2)
    
    # Performs Real-Valued Mutation
    for i, val in enumerate(updated_chrom_pop):
        if np.random.rand() <= mut_rate:
            # Adds Normal noise
            updated_chrom_pop[i] = (val + np.random.normal(0, np.std(updated_chrom_pop)))
        else:
            pass
        
    # Creates a new pop with updated weights
    for i, val in enumerate(pop):
        val.set_weights(updated_chrom_pop[i])

    # Returns the new generation
    return pop

def geneticAlg(pop: list, cross_rate: float, mut_rate: float):
    fitness = list()
    
    # Calculates the fitness of each network, depending on problem type
    for i in range(len(pop)):
        fitness.append(np.mean(pop[i].get_fitness()))
    
    best = max(fitness)
    fitness.clear()

    # Runs through an epoch, finds the best
    offspring = geneticAlgEpoch(pop, cross_rate, mut_rate)
    for i in range(len(offspring)):
        fitness.append(np.mean(offspring[i].get_fitness()))
    best_off = max(fitness)

    counter = 0
    # Continues until the two have converged
    while (np.abs(best - best_off) / best_off <= 0.05):
        if counter == 100: 
            break
        counter+=1
        
        best = best_off
        fitness.clear()

        offspring = geneticAlgEpoch(pop, cross_rate, mut_rate)
        for i in range(len(offspring)):
            fitness.append(np.mean(offspring[i].get_fitness()))
        best_off = max(fitness)

    # Returns the best offspring and its fitness score
    index = fitness.index(best_off)
    return offspring[index], best_off
    
