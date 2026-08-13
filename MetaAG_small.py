import numpy as np
import random
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from Others.Data_processor import data_processor
from Others.AG_Strategy import *
from Others.Individual import Individual

class MetaAg_small:
    def __init__(self,selection:Selection_methods,crossover:Crossover_methods,mutation:Mutation_methods):
        self.selection_strategy = selection
        self.crossover_strategy = crossover
        self.mutation_strategy = mutation

    @staticmethod
    def genome_generator(individual_lenght):
        genome = np.array([])

        for i in range(individual_lenght):
            genome = np.append(genome,random.random())
        return(genome)
    
    @staticmethod
    def determinator(tiny_list,individual):
        choice = random.choices(tiny_list, weights=individual, k=1)[0]
        return(choice)

    def create_initial_population(self,population_size,individual_lenght,matrix,X_train_transformed,a,b,g):
        population = []
        for _ in range(population_size):
            genome = self.genome_generator(individual_lenght)
            individual = Individual(genome=genome)
            fitness = self.fitness_function(individual, matrix, X_train_transformed,a,b,g)
            individual.fitness = fitness
            population.append(individual)

        # for individual in population:
        #     print(individual)
            
        return population

    def fitness_function(self,individual,matrix,X_train_transformed,a,b,g):
        matrix_ = matrix.copy()
        df_ = pd.DataFrame()
        cont = 0
        saved_dc_idx = []
        saved_cc_idx = []
        dc_sum = 0
        cc_sum = 0
        # print("esse eh o fitnessfunc")
        # print(individual)

        genome = individual.genome
        #print(genome)
        #print(matrix_)
        #choices = np.array([])
        choices = np.array([self.determinator(row, genome[:len(row)]) for row in matrix_])
        #print(choices)

        for index,values in enumerate(matrix_):
            if len(values) > 1:
                matrix_[index] = choices[cont]
                cont+=1
                saved_dc_idx.append((index,matrix_[index]))
            else:
                saved_cc_idx.append((index,matrix_[index]))

        matrix_ = [item[0] if isinstance(item, list) else item for item in matrix_]

        #naive_bayes_mult.fit(X_train_transformed,choices)
        naive_bayes_mult.partial_fit(X_train_transformed, matrix_, classes=labels)
        df_ = naive_bayes_mult.predict_log_proba(X_train_transformed)

        #print(df_)

        for index,values in saved_dc_idx:
            dc_sum += df_[index,values]

        for index,values in saved_cc_idx:
            cc_sum += df_[index,values]

        #print(matrix)
        cont = np.bincount(choices)
        variance = np.var(cont)

        #list_before_sum = [max(i) for i in df_]
        #fitness = sum(list_before_sum)

        fitness = a*cc_sum+b*dc_sum-g*variance
        #print(fitness[0])
        return(fitness)


    def genetic_algorithm(self,population_size,generations, mutation_rate,individual_lenght,matrix,X_train_transformed,a,b,g):
        population = self.create_initial_population(population_size,individual_lenght,matrix,X_train_transformed,a,b,g)
        # for individual in population:
        #     print(individual)
        # print("esse eh oo geneticalgo")
        for generation in range(generations):
            new_population = []

            best_individual = max(population, key=lambda x: self.fitness_function(x, matrix, X_train_transformed,a,b,g))
            best_fitness = self.fitness_function(best_individual, matrix, X_train_transformed,a,b,g)

            #print(best_individual.genome)
            #print(best_individual.fitness)

            index_best_individuals = sorted(range(len(population)), key=lambda x: population[x].fitness)
            primary_best_individual = population[index_best_individuals[0]]
            second_best_individual = population[index_best_individuals[1]]

            for _ in range(int((population_size-2)/2)):

                selected_crossover = self.selection_strategy.selection(population)
        
                child_1, child_2 = self.crossover_strategy.crossover(selected_crossover[0], selected_crossover[1])

                result = random.random()
                if result > mutation_rate:
                    child_1 = self.mutation_strategy.mutation(child_1)
                    child_2 = self.mutation_strategy.mutation(child_2)

                child_1.fitness = self.fitness_function(child_1, matrix, X_train_transformed, a, b, g)
                child_2.fitness = self.fitness_function(child_2, matrix, X_train_transformed, a, b, g)
                
                new_population.append(child_1)
                new_population.append(child_2)

            new_population.append(primary_best_individual)
            new_population.append(second_best_individual)

            population = new_population
            # for individual in population:
            #     print(individual)
            print(generation)

        return (best_individual)


path = "data_base/new_data_bases/bbc_text_cls_adapted_dependent_transformed_10.csv"

X_train_transformed,y_train,X_test_transformed,y_test,labels,individual_lenght,matrix,le,flag = data_processor(path,True,True)

naive_bayes_mult = MultinomialNB()

a = 1
b = 1
g = 1

Ag = MetaAg_small(selection=RankSelection(),crossover=PadronizedCrossover(),mutation=ScrambleMutation())
                   #genetic_algorithm(self,population_size,generations, mutation_rate,labels,len_,matrix,X_train_transformed,a,b,g)
best_individual =Ag.genetic_algorithm(20,10,1,individual_lenght,matrix,X_train_transformed,a,b,g)

#genome = le.inverse_transform(best_individual.genome)

# %%
#df_ = pd.DataFrame()

choices = np.array([Ag.determinator(row, best_individual.genome[:len(row)]) for row in matrix])

# if (matrix.dtype == str) or (matrix.dtype == object):
#     matrix_ = le.transform(matrix)

# if (y_train.dtype == str) or (y_train.dtype == object):
#      y_train = le.transform(y_train)

# print(y_train)
# print(choices)

# %%
print(confusion_matrix(y_train, choices))  