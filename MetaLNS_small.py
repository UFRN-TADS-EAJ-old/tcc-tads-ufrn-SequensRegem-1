import pandas as pd
import numpy as np
import random
import math
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from Others.Data_processor import data_processor
from Others.LNS_Strategy import *

class MetaLNS_small:
    def __init__(self, insertion:Insertion_methods, removal:Removal_methods):
        self.insertion_strategy = insertion
        self.removal_strategy = removal

    @staticmethod
    def determinator(tiny_list,individual):
        choice = random.choices(tiny_list, weights=individual, k=1)[0]
        return(choice)

    def fitness_function(self,individual,X_train_transformed,matrix,a,b,g):
        matrix_ = matrix.copy()
        df_ = pd.DataFrame()
        cont = 0
        saved_dc_idx = []
        saved_cc_idx = []
        dc_sum = 0
        cc_sum = 0
        # print("esse eh o fitnessfunc")
        # print(individual)

        genome = individual.copy()
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

        naive_bayes_mult.fit(X_train_transformed,choices)
        naive_bayes_mult.predict(X_train_transformed)
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
        return(fitness[0])

    def large_neigborhood_search(self,individual,n_remove,X_train_transformed,matrix,generations,labels_trans,a,b,g):
        best_fitness = float('-inf')
        best_individual = []
        fitness = 0
        
        for generation in range(generations):
            indexes = self.removal_strategy.removal(individual,n_remove)
            individual = self.insertion_strategy.insertion(individual,indexes,labels_trans)
            fitness = self.fitness_function(individual,X_train_transformed,matrix,a,b,g)
            # print(fitness)
            # print(best_fitness)
            
            if fitness > best_fitness:
                best_individual = individual
                best_fitness = fitness
            else:
                if math.exp(fitness - best_fitness) > random.random():
                    best_individual = individual
                    best_fitness = fitness

            print(generation)

        return(best_fitness,best_individual)

path = "data_base/new_data_bases/bbc_text_cls_adapted_independent_transformed_10.csv"

individual,X_train_transformed,y_train,X_test_transformed,y_test,matrix,labels_trans,le = data_processor(path,False,False)

naive_bayes_mult = MultinomialNB()

a = 1
b = 1
g = 1

LNS = MetaLNS_small(insertion=ModaInsertion(),removal=ClientProximityRemoval())

best_fitness,best_individual = LNS.large_neigborhood_search(individual,3,X_train_transformed,matrix,2000,labels_trans,a,b,g)

choices = np.array([LNS.determinator(row, best_individual[:len(row)]) for row in matrix])

print(confusion_matrix(y_train, choices))  