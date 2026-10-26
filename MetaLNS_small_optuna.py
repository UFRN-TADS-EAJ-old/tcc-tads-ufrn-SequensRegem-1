import optuna
import pandas as pd
import numpy as np
import random
import math
from sklearn.naive_bayes import MultinomialNB
from Others.Data_processor import data_processor
from Others.LNS_Strategy import *
from Save_results import Save_results

optuna.logging.set_verbosity(optuna.logging.INFO)

class MetaLNS_small:
    def __init__(self, insertion: Insertion_methods, removal: Removal_methods, matrix, mask, y_train):
        self.insertion_strategy = insertion
        self.removal_strategy = removal
        self.cc_indexes = []
        self.dc_indexes = []

        self.cc_indexes = np.argwhere(mask[y_train.index]).flatten()
        self.dc_indexes = np.argwhere(~mask[y_train.index]).flatten()

        self.cc_values = matrix[self.cc_indexes, 0].astype(int)

        self.matrix_base = np.zeros(len(matrix), dtype=int)
        if len(self.cc_indexes) > 0:
            self.matrix_base[self.cc_indexes] = self.cc_values

    @staticmethod
    def determinator(tiny_list, individual):
        if np.sum(individual) == 0:
            choice = random.choices(tiny_list, k=1)[0]
            return choice

        choice = random.choices(tiny_list, weights=individual, k=1)[0]
        return choice

    def fitness_function(self, individual, X_train_transformed, matrix, a, b, g):
        # Aqui alteramos para usar 'matrix' e fazemos a cópia corretamente
        matrix_ = matrix.copy()
        matrix_2 = self.matrix_base.copy()
        df_ = pd.DataFrame()

        choices = np.array([self.determinator(row, individual[:len(row)]) for row in matrix_])
        choices = choices[self.dc_indexes]
        matrix_2[self.dc_indexes] = choices

        naive_bayes_mult.fit(X_train_transformed, matrix_2)
        df_ = naive_bayes_mult.predict_log_proba(X_train_transformed)

        dc_sum = np.sum(df_[self.dc_indexes, choices])
        cc_sum = np.sum(df_[self.cc_indexes, self.cc_values])

        cont = np.bincount(choices)
        variance = np.var(cont)

        fitness = a * cc_sum + b * dc_sum - g * variance
        return fitness

    def large_neigborhood_search(self, individual, n_remove, X_train_transformed, matrix, generations, labels_trans, a, b, g, stop_limit, T_s, T_f):
        best_fitness = float('-inf')
        best_individual = []
        fitness = 0
        cont = 0 
        
        for generation in range(generations):
            indexes = self.removal_strategy.removal(individual, n_remove)
            individual = self.insertion_strategy.insertion(individual, indexes, labels_trans)
            fitness = self.fitness_function(individual, X_train_transformed, matrix, a, b, g)
            mi = (T_f / T_s) ** (1 / (generations - 1))
            T_moment = T_s * mi ** generation
            
            if fitness > best_fitness:
                best_individual = individual.copy()
                best_fitness = fitness
                cont = 0
            else:
                cont += 1
                delta = fitness - best_fitness
                if math.exp(delta / T_moment) > random.random():
                    best_individual = individual
                    best_fitness = fitness

            if cont == stop_limit:
                break

        return best_fitness, best_individual

naive_bayes_mult = MultinomialNB()

def objective(trial):
    # Otimização dos parâmetros a, b e g (inteiros de 0 a 10)
    a = trial.suggest_int('a', 0, 10)
    b = trial.suggest_int('b', 0, 10)
    g = trial.suggest_int('g', 0, 10)
    
    # Otimização da taxa de infiltração
    percentage = trial.suggest_categorical('percentage', [1, 5, 10, 15, 20])
    
    name = "SMSSpamCollection_adapted_dependent_transformed_"
    path = f"data_base/new_data_bases/{name}{percentage}.csv"

    individual, X_train_transformed, y_train, X_test_transformed, y_test, matrix, labels_trans, le, mask_cc = data_processor(path, False)

    insertion_choice = trial.suggest_categorical('insertion_method', ['moda', 'random'])
    removal_choice = trial.suggest_categorical('removal_method', ['sequence', 'proximity'])

    if insertion_choice == 'moda':
        insertion_method = ModaInsertion_small(columns=individual)
    else:
        insertion_method = RandomInsertion_small()

    if removal_choice == 'sequence':
        removal_method = SequenceRemoval()
    else:
        removal_method = ClienteProximityRemoval_small()

    LNS = MetaLNS_small(insertion=insertion_method, removal=removal_method, matrix=matrix, mask=mask_cc, y_train=y_train)
    best_fitness, best_individual = LNS.large_neigborhood_search(
        individual, 3, X_train_transformed, matrix, 1000, labels_trans, a, b, g, 30, 2, 0.1
    )
    
    return best_fitness

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    # n_jobs=2 roda 2 trials simultaneamente, seguro para 8GB de RAM no Windows 11
    study.optimize(objective, n_trials=100, n_jobs=2)

    print("\n--- RESULTADOS DA OTIMIZAÇÃO (SMALL) ---")
    print(f"Melhor Fitness: {study.best_value}")
    print("Melhores Hiperparâmetros Encontrados:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")