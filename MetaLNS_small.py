import pandas as pd
import numpy as np
import random
import math
from sklearn.naive_bayes import MultinomialNB
from Others.Data_processor import data_processor
from Others.LNS_Strategy import *
from Save_results import Save_results

class MetaLNS_small:
    def __init__(self, insertion:Insertion_methods, removal:Removal_methods, matrix):
        self.insertion_strategy = insertion
        self.removal_strategy = removal
        self.cc_indexes = []
        self.dc_indexes = []

        for index,values in enumerate(matrix):
            if len(set(values)) > 1:
                self.dc_indexes.append(index)
            else:
                self.cc_indexes.append(index)


    @staticmethod
    def determinator(tiny_list,individual):
        if np.sum(individual) == 0:
            choice = random.choices(tiny_list, k=1)[0]
            return(choice)

        choice = random.choices(tiny_list, weights=individual, k=1)[0]
        return(choice)

    def fitness_function(self,individual,X_train_transformed,matrix,a,b,g):
        matrix_ = matrix.copy()
        df_ = pd.DataFrame()
        cont = 0

        choices = np.array([self.determinator(row, individual[:len(row)]) for row in matrix_])

       # matrix_ = [item[0] if isinstance(item, list) else item for item in matrix_]

        naive_bayes_mult.fit(X_train_transformed,choices)
        # naive_bayes_mult.predict(X_train_transformed)
        df_ = naive_bayes_mult.predict_log_proba(X_train_transformed)

        #print(df_)

        dc_sum = np.sum(np.fromiter((df_[idx, choices[idx]] for idx in self.dc_indexes),dtype=int))

        cc_sum = np.sum(np.fromiter((df_[idx, matrix[idx][0]] for idx in self.cc_indexes),dtype=int))

        #print(matrix)
        #olhar!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        cont = np.bincount(choices)
        variance = np.var(cont)
        variance = np.array(variance)

        #list_before_sum = [max(i) for i in df_]
        #fitness = sum(list_before_sum)

        fitness = a*cc_sum+b*dc_sum-g*variance
        # print(fitness)
        return(fitness)

    def large_neigborhood_search(self,individual,n_remove,X_train_transformed,matrix,generations,labels_trans,a,b,g,stop_limit):
        best_fitness = float('-inf')
        best_individual = []
        fitness = 0
        cont = 0 
        
        for generation in range(generations):
            indexes = self.removal_strategy.removal(individual,n_remove)
            individual = self.insertion_strategy.insertion(individual,indexes,labels_trans)
            fitness = self.fitness_function(individual,X_train_transformed,matrix,a,b,g)
            # print(fitness)
            # print(best_fitness)
            
            if fitness > best_fitness:
                best_individual = individual.copy()
                best_fitness = fitness
                cont = 0
            else:
                cont += 1
                if math.exp(fitness - best_fitness) > random.random():
                    best_individual = individual
                    best_fitness = fitness

            print(generation)
            if cont == stop_limit:
                break

        return(best_fitness,best_individual)

naive_bayes_mult = MultinomialNB()

a = 5
b = 1
g = 0.5

percentages = [1,5,10,15,20]

data_names = [
    # "bbc_text_cls_adapted_dependent_transformed_",
    # "cyberbullying_tweets_adapted_dependent_transformed_",
    # "fake_news_train_adapted_dependent_transformed_",
    # "data_for_AI_adapted_dependent_transformed_", #Novo
    # "humor_classification_dataset_adapted_dependent_transformed_", #
    "Emotion_classify_Data_adapted_dependent_transformed_", #Novo
    # "News_Category_Dataset_v3_adapted_dependent_transformed_", #
    "SMSSpamCollection_adapted_dependent_transformed_",
    # "train_SST2_dependent_transformed_",#
    "Emotion_classification_adapted_dependent_transformed_",
    # "Text_classifi1cation_documents_df_file_adapted_dependent_transformed_", #Novo
    # "train_yelp_dependent_transformed_", #
    "IMDB_Dataset_adapted_dependent_transformed_",
    "Womens_Clothing_E-Commerce_Reviews_adapted_dependent_transformed_"
]


Removal_methods_list = [
    SequenceRemoval(),
    ClienteProximityRemoval_small()
]

for name in data_names:
    for percentage in percentages:

        path = f"data_base/new_data_bases/{name}{percentage}.csv"

        individual,X_train_transformed,y_train,X_test_transformed,y_test,matrix,labels_trans,le = data_processor(path,False,False)

        Insertion_methods_list = [
            ModaInsertion_small(columns=individual),
            RandomInsertion_small(),
        ]

        for insertion_method in Insertion_methods_list:
            for removal_method in Removal_methods_list:
                for _ in range(30):

                    #LNS = MetaLNS_small(insertion=ModaInsertion(columns=individual),removal=SequenceRemoval())
                    LNS = MetaLNS_small(insertion=insertion_method,removal=removal_method,matrix=matrix)

                    best_fitness,best_individual = LNS.large_neigborhood_search(individual,3,X_train_transformed,matrix,2000,labels_trans,a,b,g,30)

                    choices = np.array([LNS.determinator(row, best_individual[:len(row)]) for row in matrix])

                    #print(confusion_matrix(y_train, choices))

                    Save_results(type=False,sub_type=False,data_name=path,y_true=y_train,y_pred=choices,methods=(insertion_method,removal_method),a=a,b=b,g=g)