from abc import ABC, abstractmethod
import random
import numpy as np
from Others.Individual import Individual

class Selection_methods(ABC):
    @abstractmethod
    def selection (population):
        pass

class Crossover_methods(ABC):
    @abstractmethod
    def crossover (individual_1,individual_2):
        pass

class Mutation_methods(ABC):
    @abstractmethod
    def mutation (individual):
        pass

class TournmentSelection(Selection_methods):
     def selection (self,population) -> list:
        individuals = []
        available = range(len(population))
        sorted_parents = random.sample(available,4)
        parent1 = population[sorted_parents[0]]
        parent2 = population[sorted_parents[1]]
        parent3 = population[sorted_parents[2]]
        parent4 = population[sorted_parents[3]]

        winner_1 = max(parent1,parent2, key=lambda x: x.fitness)
        winner_2 = max(parent3,parent4, key=lambda x: x.fitness)
        individuals.append(winner_1)
        individuals.append(winner_2)
        return(individuals)

class StochastictSelection(Selection_methods):
     def selection (self,population) -> list:
        sum = 0
        sum_2 = 0
        n_select = 10
        cont = 0
        spaces = []
        individuals = []

        smallest_fitness = min(individual.fitness for individual in population)

        offset = abs(smallest_fitness) if smallest_fitness < 0 else 0

        for individual in population:
            sum += individual.fitness + offset

        space = sum / n_select if n_select > 0 else 0
        if space > 0:
            spaces.append(random.randint(0, int(space)))
            for i in range(0, 9):  # 9 iterações para completar 10 elementos no total
                spaces.append(spaces[i] + space)
        else:
            spaces = [0] * n_select

        current_sum = 0
        for individual in population:
            current_sum += individual.fitness + offset

            # Enquanto o fitness acumulado atingir o espaço do ponteiro atual
            while cont < len(spaces) and current_sum >= spaces[cont]:
                individuals.append(individual)
                cont += 1
                if cont >= len(spaces):
                    break

        if len(individuals)%2 == 1:
            to_remove = random.choice(individuals)
            individuals.remove(to_remove)

        return(individuals)

class RankSelection(Selection_methods):
     def selection (self,population) -> list:
        list_ = []
        sorted_ascending = sorted(population, key=lambda individual: individual.fitness, reverse=True)
        len_ = len(sorted_ascending)
        total = ((len_*(len_+1))/2)

        for i in range(len_):
            list_.append((i+1)/total)

        individuals = random.choices(sorted_ascending, weights = list_, k = 4)
        return(individuals)

class SiglePointCrossover(Crossover_methods):
    def crossover (self,individual_1,individual_2) -> Individual:
        parental_genome_1 = individual_1.genome
        parental_genome_2 = individual_2.genome
 
        cut = random.randint(1,len(parental_genome_1))
        part_1 = parental_genome_1[:cut]
        part_2 = parental_genome_2[cut:]

        genome_1 = np.concatenate((part_1, part_2))
        genome_2 = np.concatenate((part_2, part_1))

        child_1 = Individual(genome=genome_1)
        child_2 = Individual(genome=genome_2)

        return(child_1,child_2)

class TwoPointCrossover(Crossover_methods):
    def crossover (self,individual_1,individual_2) -> Individual:
        parental_genome_1 = individual_1.genome
        parental_genome_2 = individual_2.genome
        cut_1 = random.randint(1,len(parental_genome_1))
        try:
            cut_2 = cut_1+20
            parental_genome_1[cut_2]
        except IndexError:
            cut_2 = cut_1-20

        if cut_1 < cut_2:
            part_1 = parental_genome_1[:cut_1]
            part_2 = parental_genome_1[cut_1:cut_2]
            part_3 = parental_genome_1[cut_2:]
            part_4 = parental_genome_2[:cut_1]
            part_5 = parental_genome_2[cut_1:cut_2]
            part_6 = parental_genome_2[cut_2:]
        else:
            part_1 = parental_genome_1[:cut_2]
            part_2 = parental_genome_1[cut_2:cut_1]
            part_3 = parental_genome_1[cut_1:]
            part_4 = parental_genome_2[:cut_2]
            part_5 = parental_genome_2[cut_2:cut_1]
            part_6 = parental_genome_2[cut_1:]

        genome_1 = np.concatenate((part_1, part_5, part_3))
        genome_2 = np.concatenate((part_4, part_2, part_6))

        child_1 = Individual(genome=genome_1)
        child_2 = Individual(genome=genome_2)
        
        return(child_1,child_2)

class PadronizedCrossover(Crossover_methods):
    def crossover (self,individual_1,individual_2) -> Individual:
        genome_1 = []
        genome_2 = []
        parental_genome_1 = individual_1.genome
        parental_genome_2 = individual_2.genome
        if len(parental_genome_1)%2 == 0:
            list_ = [0,1]*int((len(parental_genome_1)/2))
        else:
            result = len(parental_genome_1)/2
            result = -(-result // 1)
            list_ = [0,1]*result
            list_ = list_.pop()
        for i in range(len(parental_genome_1)):
            if list_[i] == 0:
                genome_1.append(parental_genome_1[i])
                genome_2.append(parental_genome_2[i])
            else:
                genome_1.append(parental_genome_2[i])
                genome_2.append(parental_genome_1[i])

        child_1 = Individual(genome=genome_1)
        child_2 = Individual(genome=genome_2)
        
        return(child_1,child_2)

class DisplacementMutation(Mutation_methods):    
    def mutation (self,child) -> Individual:
        genome = child.genome
        cut_1 = random.randint(1,len(genome))
        try:
            cut_2 = cut_1+20
            genome[cut_2]
        except IndexError:
            cut_2 = cut_1-20

        if cut_1 < cut_2:
            part_1 = genome[:cut_1]
            part_2 = genome[cut_1:cut_2]
            part_3 = genome[cut_2:]
        else:
            part_1 = genome[:cut_2]
            part_2 = genome[cut_2:cut_1]
            part_3 = genome[cut_1:]

        genome = np.concatenate((part_1, part_3, part_2))
        child.genome = genome

        return(child)

class SimpleInversionMutation(Mutation_methods):    
    def mutation (self,child) -> Individual:
        genome = child.genome
        cut_1 = random.randint(1,len(genome))
        try:
            cut_2 = cut_1+20
            genome[cut_2]
        except IndexError:
            cut_2 = cut_1-20

        if cut_1 < cut_2:
            part_1 = genome[:cut_1]
            part_2 = genome[cut_1:cut_2]
            part_3 = genome[cut_2:]
        else:
            part_1 = genome[:cut_2]
            part_2 = genome[cut_2:cut_1]
            part_3 = genome[cut_1:]

        part_2 = part_2[::-1]
        
        genome = np.concatenate((part_1, part_2, part_3))
        child.genome = genome

        return(child)

class ScrambleMutation(Mutation_methods):    
    def mutation (self,child) -> Individual:
        genome = child.genome
        cut_1 = random.randint(1,len(genome))
        try:
            cut_2 = cut_1+20
            genome[cut_2]
        except IndexError:
            cut_2 = cut_1-20

        if cut_1 < cut_2:
            part_1 = genome[:cut_1]
            part_2 = genome[cut_1:cut_2]
            part_3 = genome[cut_2:]
        else:
            part_1 = genome[:cut_2]
            part_2 = genome[cut_2:cut_1]
            part_3 = genome[cut_1:]

        random.shuffle(part_2)

        genome = np.concatenate((part_1, part_2, part_3))

        child.genome = genome

        return(child)