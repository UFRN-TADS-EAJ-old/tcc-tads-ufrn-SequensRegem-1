import pandas as pd
import numpy as np
import random
import math
from sklearn.naive_bayes import MultinomialNB
from Others.Data_processor import data_processor
from Others.LNS_Strategy import *
from Save_results import Save_results

class MetaLNS_big:
    def __init__(self, insertion:Insertion_methods, removal:Removal_methods, matrix, mask, y_train):
        self.insertion_strategy = insertion
        self.removal_strategy = removal
        self.cc_indexes = []
        self.dc_indexes = []

        # for index,values in enumerate(matrix):
        #     if len(set(values)) > 1:
        #         self.dc_indexes.append(index)
        #     else:
        #         self.cc_indexes.append(index)

        self.cc_indexes = np.argwhere(mask[y_train.index]).flatten()
        self.dc_indexes = np.argwhere(~mask[y_train.index]).flatten()

        # self.cc_values = np.array([matrix[i][0] if isinstance(matrix[i], list) else matrix[i] for i in self.cc_indexes], dtype=int)
        self.cc_values = matrix[self.cc_indexes].astype(int)

        self.matrix_base = np.zeros(len(matrix), dtype=int)
        if len(self.cc_indexes) > 0:
            self.matrix_base[self.cc_indexes] = self.cc_values


    def fitness_function(self,individual,X_train_transformed,a,b,g):
        matrix_ = self.matrix_base.copy()
        df_ = pd.DataFrame()
        # individual_arr = np.asarray(individual, dtype=int)
        #matrix_ = [item[0] if isinstance(item, list) else item for item in matrix_]
        matrix_[self.dc_indexes] = individual
        # matrix_[self.cc_indexes] = self.cc_values

        #matrix_2 = np.array([item[0] if isinstance(item, (list, np.ndarray)) else item for item in matrix_], dtype=int)

        # naive_bayes_mult.fit(X_train_transformed,matrix_)
        naive_bayes_mult.fit(X_train_transformed, matrix_)
        df_ = naive_bayes_mult.predict_log_proba(X_train_transformed)

        dc_sum = np.sum(df_[self.dc_indexes, individual])
        cc_sum = np.sum(df_[self.cc_indexes, self.cc_values])

        cont = np.bincount(matrix_)
        variance = np.var(cont)

        #list_before_sum = [max(i) for i in df_]
        #fitness = sum(list_before_sum)

        fitness = a*cc_sum+b*dc_sum-g*variance
        
        return(fitness)

    def large_neigborhood_search(self,individual,n_remove,X_train_transformed,matrix,generations,labels_trans,a,b,g,stop_limit,T_s,T_f):
        best_fitness = float('-inf')
        best_individual = []
        fitness = 0
        cont = 0
        
        for generation in range(generations):
            indexes = self.removal_strategy.removal(individual,n_remove)
            individual = self.insertion_strategy.insertion(individual,indexes,labels_trans)
            fitness = self.fitness_function(individual,X_train_transformed,a,b,g)
            mi = (T_f/T_s)**(1/(generations-1))
            T_moment = T_s*mi**generation
            # print(fitness)
            # print(best_fitness)
            
            if fitness > best_fitness:
                best_individual = individual.copy()
                best_fitness = fitness
                cont=0
            else:
                cont +=1
                delta = fitness - best_fitness
                if math.exp(delta/T_moment) > random.random():
                    best_individual = individual
                    best_fitness = fitness

            print(generation)
            if cont == stop_limit:
                break

        return(best_fitness,best_individual)




a = 5
b = 1
g = 0.5

naive_bayes_mult = MultinomialNB()


percentages = [1,5,10,15,20]

data_names = [
    # "bbc_text_cls_adapted_dependent_transformed_",
    # "cyberbullying_tweets_adapted_dependent_transformed_",
    # "fake_news_train_adapted_dependent_transformed_",
    # "data_for_AI_adapted_dependent_transformed_", #Novo
    # # "humor_classification_dataset_adapted_dependent_transformed_", #
    # "Emotion_classify_Data_adapted_dependent_transformed_", #Novo
    # # "News_Category_Dataset_v3_adapted_dependent_transformed_", #
    "SMSSpamCollection_adapted_dependent_transformed_",
    # "train_SST2_dependent_transformed_",#
    # "Emotion_classification_adapted_dependent_transformed_",
    # # "Text_classifi1cation_documents_df_file_adapted_dependent_transformed_", #Novo
    # # "train_yelp_dependent_transformed_", #
    # "IMDB_Dataset_adapted_dependent_transformed_",
    # "Womens_Clothing_E-Commerce_Reviews_adapted_dependent_transformed_"
]



Removal_methods_list = [
    RouteRemoval(),
    SequenceRemoval()
]

for name in data_names:
    for percentage in percentages:
            path = f"data_base/new_data_bases/{name}{percentage}.csv"

            individual,X_train_transformed,y_train,X_test_transformed,y_test,matrix,labels_trans,le, mask_cc = data_processor(path=path,big_or_small=True)

            Insertion_methods_list = [
                ModaInsertion_big(columns=matrix[np.argwhere(~mask_cc[y_train.index]).flatten()]),
                RandomInsertion_big(),
            ]

            # matrix_2 = matrix.copy()
            
            for insertion_method in Insertion_methods_list:
                for removal_method in Removal_methods_list:
                    # for _ in range(30):

                        LNS = MetaLNS_big(insertion=insertion_method,removal=removal_method,matrix=matrix,mask=mask_cc,y_train=y_train)

                        best_fitness,best_individual = LNS.large_neigborhood_search(individual,3,X_train_transformed,matrix,2000,labels_trans,a,b,g,30,2,0.1)

                        cont = 0
                        # for index,values in enumerate(matrix_2):
                        #     if len(values) > 1:
                        #         matrix_2[index] = best_individual[cont]
                        #         cont+=1

                        # matrix_2 = [item[0] if isinstance(item, list) else item for item in matrix_2]
                        matrix = [item[0] if isinstance(item, list) else item for item in matrix]
                        # Save_results(type=False,sub_type=True,data_name=path,y_true=y_train,y_pred=matrix_2,methods=(insertion_method,removal_method),a=a,b=b,g=g)
