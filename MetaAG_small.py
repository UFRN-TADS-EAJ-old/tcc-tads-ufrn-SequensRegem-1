import numpy as np
import random
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from Others.Data_processor import data_processor
from Others.AG_Strategy import *
from Others.Individual import Individual
from Save_results import Save_results
import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed

class MetaAg_small:
    def __init__(self,selection:Selection_methods,crossover:Crossover_methods,mutation:Mutation_methods,matrix):
        self.selection_strategy = selection
        self.crossover_strategy = crossover
        self.mutation_strategy = mutation

        self.cc_indexes = []
        self.dc_indexes = []
        self.cc_values = []

        for index,values in enumerate(matrix):
            if len(set(values)) > 1:
                self.dc_indexes.append(index)
            else:
                self.cc_indexes.append(index)
                self.cc_values.append(values[0])

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

    def create_initial_population(self,population_size,individual_lenght,matrix,X_train_transformed,a,b,g, naive_bayes_mult):
        population = []
        for _ in range(population_size):
            genome = self.genome_generator(individual_lenght)
            individual = Individual(genome=genome)
            fitness = self.fitness_function(individual, matrix, X_train_transformed,a,b,g, naive_bayes_mult)
            individual.fitness = fitness
            population.append(individual)

        # for individual in population:
        #     print(individual)
            
        return population

    def fitness_function(self,individual,matrix,X_train_transformed,a,b,g, naive_bayes_mult):
        genome = individual.genome

        y = np.empty(len(matrix), dtype=int)

        if len(self.cc_indexes) > 0:
            y[self.cc_indexes] = self.cc_values

        #choices = np.array([self.determinator(row, genome[:len(row)]) for row in matrix[self.dc_indexes]])

        matrix_dc = [matrix[i] for i in self.dc_indexes]
        choices = np.array([self.determinator(row, genome[:len(row)]) for row in matrix_dc])

        if len(self.dc_indexes) > 0:
            y[self.dc_indexes] = choices

        naive_bayes_mult.fit(X_train_transformed, y)
        df_ = naive_bayes_mult.predict_log_proba(X_train_transformed)

        dc_sum = np.sum(df_[self.dc_indexes, y[self.dc_indexes]])
        cc_sum = np.sum(df_[self.cc_indexes, y[self.cc_indexes]])

        # dc_sum = np.sum(np.fromiter((df_[idx, choices[idx]] for idx in self.dc_indexes),dtype=int))
        # cc_sum = np.sum(np.fromiter((df_[idx, matrix[idx][0]] for idx in self.cc_indexes),dtype=int))

        #print(matrix)
        cont = np.bincount(y)
        variance = np.var(cont)

        #list_before_sum = [max(i) for i in df_]
        #fitness = sum(list_before_sum)

        fitness = a*cc_sum+b*dc_sum-g*variance
        #print(fitness[0])
        return(fitness)


    def genetic_algorithm(self,population_size,generations, mutation_rate,individual_lenght,matrix,X_train_transformed,a,b,g,stop_limit, naive_bayes_mult):
        population = self.create_initial_population(population_size,individual_lenght,matrix,X_train_transformed,a,b,g, naive_bayes_mult)
        best_individual_ = None
        best_fitness_ = float('-inf')
        cont = 0
        # for individual in population:
        #     print(individual)
        # print("esse eh oo geneticalgo")
        for generation in range(generations):
            new_population = []

            # best_individual = max(population, key=lambda x: self.fitness_function(x, matrix, X_train_transformed,a,b,g))
            # best_fitness = self.fitness_function(best_individual, matrix, X_train_transformed,a,b,g)

            population.sort(key=lambda x: x.fitness)
            primary_best_individual = population[-1]
            second_best_individual = population[-2]

            best_individual = primary_best_individual
            best_fitness = best_individual.fitness

            if  best_fitness == None or best_fitness > best_fitness_:
                best_individual_ = best_individual
                cont = 0
            else:
                cont += 1

            if cont == stop_limit:
                break

            for _ in range(int((population_size-2)/2)):

                selected_crossover = self.selection_strategy.selection(population)
        
                child_1, child_2 = self.crossover_strategy.crossover(selected_crossover[0], selected_crossover[1])

                result = random.random()
                if result < mutation_rate:
                    child_1 = self.mutation_strategy.mutation(child_1)
                    child_2 = self.mutation_strategy.mutation(child_2)

                child_1.fitness = self.fitness_function(child_1, matrix, X_train_transformed, a, b, g, naive_bayes_mult)
                child_2.fitness = self.fitness_function(child_2, matrix, X_train_transformed, a, b, g, naive_bayes_mult)
                
                new_population.extend([child_1,child_2])

            new_population.extend([primary_best_individual,second_best_individual])

            population = new_population
            # for individual in population:
            #     print(individual)
            print(generation)

        return (best_individual_)


#naive_bayes_mult = MultinomialNB()

# a = 5
# b = 1
# g = 0.5

# percentages = [1,5,10,15,20]

# data_names = [
#     "bbc_text_cls_adapted_dependent_transformed_",
#     "cyberbullying_tweets_adapted_dependent_transformed_",
#     "fake_news_train_adapted_dependent_transformed_",
#     "data_for_AI_adapted_dependent_transformed_", #Novo
#     # "humor_classification_dataset_adapted_dependent_transformed_", #
#     "IMDB_Dataset_adapted_dependent_transformed_",
#     "Emotion_classify_Data_adapted_dependent_transformed_", #Novo
#     # "News_Category_Dataset_v3_adapted_dependent_transformed_", #
#     "SMSSpamCollection_adapted_dependent_transformed_",
#     "train_SST2_dependent_transformed_",
#     "Text_classification_documents_df_file_adapted_dependent_transformed_", #Novo
#     # "train_yelp_dependent_transformed_", #
#     "Womens_Clothing_E-Commerce_Reviews_adapted_dependent_transformed_"
# ]

# selection_methods_list = [
#     TournmentSelection(),
#     StochastictSelection(),
#     RankSelection()
# ]

# crossover_method_list = [
#     SiglePointCrossover(),
#     TwoPointCrossover(),
#     PadronizedCrossover(),
# ]

# mutatuion_method_list = [
#     DisplacementMutation(),
#     SimpleInversionMutation(),
#     ScrambleMutation()
# ]

# for name in data_names:
#     for percentage in percentages:

#         path = f"data_base/new_data_bases/{name}{percentage}.csv"

#         X_train_transformed,y_train,X_test_transformed,y_test,labels,individual_lenght,matrix,le,flag = data_processor(path,True,True)

#         for selection_method in selection_methods_list:
#             for crossover_method in crossover_method_list:
#                 for mutation_method in mutatuion_method_list:
#                     Ag = MetaAg_small(selection=selection_method,crossover=crossover_method,mutation=mutation_method)
#                                     #genetic_algorithm(self,population_size,generations, mutation_rate,labels,len_,matrix,X_train_transformed,a,b,g)
#                     best_individual = Ag.genetic_algorithm(20,1000,0.5,individual_lenght,matrix,X_train_transformed,a,b,g,30)

#                     print(best_individual)

#                     choices = np.array([Ag.determinator(row, best_individual.genome[:len(row)]) for row in matrix]) 
#                     #Save_results(type=True,sub_type=False,data_name=path,y_true=y_train,y_pred=choices,methods=(selection_method,crossover_method,mutation_method),a=a,b=b,g=g)

def run_single_experiment(args):
    """Função isolada executada em cada processo paralelo."""
    name, percentage, sel_method, cross_method, mut_method, a, b, g = args
    
    path = f"data_base/new_data_bases/{name}{percentage}.csv"
    
    X_train_transformed, y_train, X_test_transformed, y_test, labels, individual_lenght, matrix, le, flag = data_processor(path, True, True)

    # Criação do modelo local dentro do processo
    naive_bayes_mult = MultinomialNB()

    Ag = MetaAg_small(selection=sel_method, crossover=cross_method, mutation=mut_method, matrix=matrix)

    best_individual = Ag.genetic_algorithm(
        population_size=20,
        generations=1000,
        mutation_rate=0.5,
        individual_lenght=individual_lenght,
        matrix=matrix,
        X_train_transformed=X_train_transformed,
        a=a, b=b, g=g,
        stop_limit=30,
        naive_bayes_mult=naive_bayes_mult
    )

    choices = np.array([Ag.determinator(row, best_individual.genome[:len(row)]) for row in matrix]) 
    
    Save_results(
        type=True,
        sub_type=False,
        data_name=path,
        y_true=y_train,
        y_pred=choices,
        methods=(sel_method, cross_method, mut_method),
        a=a, b=b, g=g
    )

    return f"[CONCLUÍDO] {name} | {percentage}% | {sel_method.__class__.__name__} | {cross_method.__class__.__name__} | {mut_method.__class__.__name__}"


if __name__ == '__main__':
    a = 5
    b = 1
    g = 0.5

    percentages = [1, 5, 10, 15, 20]

    data_names = [
        "bbc_text_cls_adapted_dependent_transformed_",
        "cyberbullying_tweets_adapted_dependent_transformed_",
        "fake_news_train_adapted_dependent_transformed_",
        "data_for_AI_adapted_dependent_transformed_",
        "IMDB_Dataset_adapted_dependent_transformed_",
        "Emotion_classify_Data_adapted_dependent_transformed_",
        "SMSSpamCollection_adapted_dependent_transformed_",
        "train_SST2_dependent_transformed_",
        "Text_classification_documents_df_file_adapted_dependent_transformed_",
        "Womens_Clothing_E-Commerce_Reviews_adapted_dependent_transformed_"
    ]

    selection_methods_list = [
        TournmentSelection(),
        StochastictSelection(),
        RankSelection()
    ]

    crossover_method_list = [
        SiglePointCrossover(),
        TwoPointCrossover(),
        PadronizedCrossover(),
    ]

    mutatuion_method_list = [
        DisplacementMutation(),
        SimpleInversionMutation(),
        ScrambleMutation()
    ]

    # Prepara a lista com todas as combinações de experimentos
    tasks = []
    for name in data_names:
        for percentage in percentages:
            for sel, cross, mut in itertools.product(selection_methods_list, crossover_method_list, mutatuion_method_list):
                tasks.append((name, percentage, sel, cross, mut, a, b, g))

    # Define número seguro de workers (deixa 2 núcleos livres para o sistema/notebook)
    total_cores = os.cpu_count() or 4
    safe_workers = max(1, total_cores - 2)

    print(f"Iniciando paralelização com {safe_workers} processos simultâneos em {len(tasks)} experimentos...")

    with ProcessPoolExecutor(max_workers=safe_workers) as executor:
        futures = [executor.submit(run_single_experiment, task) for task in tasks]
        for future in as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                print(f"[ERRO]: {e}")