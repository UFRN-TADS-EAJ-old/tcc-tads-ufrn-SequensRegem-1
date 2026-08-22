import numpy as np
import random
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from Others.Data_processor import data_processor
from Others.AG_Strategy import *
from Others.Individual import Individual

class MetaAg_big:
    def __init__(self,selection:Selection_methods,crossover:Crossover_methods,mutation:Mutation_methods):
        self.selection_strategy = selection
        self.crossover_strategy = crossover
        self.mutation_strategy = mutation

        self.cc_indexes = []
        self.dc_indexes = []

        for index,values in enumerate(matrix):
            if len(set(values)) > 1:
                self.dc_indexes.append(index)
            else:
                self.cc_indexes.append(index)

    def create_initial_population(self,size,len_,labels,matrix,X_train_transformed,a,b,g):
        population = []
        for _ in range(size):
            genome = random.choices(labels, k=len_)
            genome = np.array(genome)
            individual = Individual(genome=genome)
            fitness = self.fitness_function(individual,matrix,X_train_transformed,a,b,g)
            individual.fitness = fitness
            #individual.fitness = fitness
            #print(individual.genome)
            population.append(individual)
            
        return population

    def fitness_function(self,individual,matrix,X_train_transformed,a,b,g):
        matrix_ = matrix.copy()
        df_ = pd.DataFrame()

        genome= np.asarray(individual.genome, dtype=int)

        matrix_[self.dc_indexes] = genome

        matrix_ = [item[0] if isinstance(item, list) else item for item in matrix_]

        naive_bayes_mult.fit(X_train_transformed,matrix_)
        # naive_bayes_mult.partial_fit(X_train_transformed, matrix_, classes=labels)
        df_ = naive_bayes_mult.predict_log_proba(X_train_transformed)

        dc_sum = np.sum(np.fromiter((df_[idx, matrix[idx]][0] for idx in self.dc_indexes),dtype=int))

        cc_sum = np.sum(np.fromiter((df_[idx, matrix[idx][0]] for idx in self.cc_indexes),dtype=int))

        cont = np.bincount(matrix_)
        variance = np.var(cont)
        variance = np.array(variance)

        #list_before_sum = [max(i) for i in df_]
        #fitness = sum(list_before_sum)

        fitness = a*cc_sum+b*dc_sum-g*variance
        #print(fitness)
        
        return(fitness)


    def genetic_algorithm(self,population_size,generations, mutation_rate,labels,len_,matrix,X_train_transformed,a,b,g,stop_limit):
        population = self.create_initial_population(population_size,len_,labels,matrix,X_train_transformed,a,b,g)
        best_individual_ = None
        best_fitness_ = float('-inf')
        cont = 0


        for generation in range(generations):
            new_population = []

            population.sort(key=lambda x: x.fitness)
            primary_best_individual = population[-1]
            second_best_individual = population[-2]

            best_individual = max(population, key=lambda x: self.fitness_function(x, matrix, X_train_transformed,a,b,g))
            best_fitness = self.fitness_function(best_individual, matrix, X_train_transformed,a,b,g)

            if best_fitness > best_fitness_:
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
            print(generation)

        return (best_individual_)


naive_bayes_mult = MultinomialNB()

a = 5
b = 1
g = 0.5

percentages = [1,5,10,15,20]

data_names = [
    "bbc_text_cls_adapted_dependent_transformed_",
    "cyberbullying_tweets_adapted_dependent_transformed_",
    "fake_news_train_adapted_dependent_transformed_",
    "data_for_AI_adapted_dependent_trnasformed_", #Novo
    # "humor_classification_dataset_adapted_dependent_transformed_", #
    "IMDB_Dataset_adapted_dependent_transformed_",
    "Emotion_classify_Data_adapted_transformed_", #Novo
    # "News_Category_Dataset_v3_adapted_dependent_transformed_", #
    "SMSSpamCollection_adapted_dependent_transformed_",
    "train_SST2_dependent_transformed_",
    "Text_classification_documents_df_file_adapted_transformed_" #Novo
    # "train_yelp_dependent_transformed_", #
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

for name in data_names:
    for percentage in percentages:

        path = f"data_base/new_data_bases/{name}{percentage}.csv"

        X_train_transformed,y_train,X_test_transformed,y_test,labels,individual_lenght,matrix,le,flag = data_processor(path,True,True)

        for selection_method in selection_methods_list:
            for crossover_method in crossover_method_list:
                for mutation_method in mutatuion_method_list:
                    Ag = MetaAg_big(selection=selection_method,crossover=crossover_method,mutation=mutation_method)
                                    #genetic_algorithm(self,population_size,generations, mutation_rate,labels,len_,matrix,X_train_transformed,a,b,g)
                    best_individual = Ag.genetic_algorithm(20,1000,0.5,labels,individual_lenght,matrix,X_train_transformed,a,b,g,30)

                    print(best_individual)

                    choices = np.array([Ag.determinator(row, best_individual.genome[:len(row)]) for row in matrix]) 
                    #Save_results(type=True,sub_type=False,data_name=path,y_true=y_train,y_pred=choices,methods=(selection_method,crossover_method,mutation_method),a=a,b=b,g=g)