"""
MetaLNS_small + Optuna em paralelo.

Rode pelo TERMINAL, na raiz do projeto (não pelo Spyder/Jupyter, que travam o multiprocessing no Windows):

    python MetaLNS_small.py
    python MetaLNS_small.py --trials 80 --workers 3 --repeticoes 2
    python MetaLNS_small.py --por-infiltracao     # um estudo (com o mesmo nº de trials) para cada infiltração
    python MetaLNS_small.py --novo                # apaga o estudo salvo e recomeça

O Optuna procura a, b, g (inteiros de 0 a 10) e a infiltração (1, 5, 10, 15, 20) que MAXIMIZAM a
qualidade dos rótulos inferidos (métrica abaixo) comparados com os rótulos reais do treino (y_train).
(Não dá para maximizar o próprio fitness do LNS: ele depende de a, b, g, e o máximo seria sempre a=b=g=0.)
"""
import os

# 1 thread de BLAS por processo: evita 3-4 workers disputando os mesmos núcleos.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import math
import random
import time

import numpy as np
from sklearn.naive_bayes import MultinomialNB

from Others.Data_processor import data_processor
from Others.LNS_Strategy import (Insertion_methods, Removal_methods, ModaInsertion_big,
                                 RandomInsertion_big, SequenceRemoval,
                                 ClientProximityRemoval_big)
from Others.Optuna_parallel import (calcular_metrica, normalizar_rotulos, relatorio,
                                    rodar_estudo, workers_seguros)


class MetaLNS_small:
    def __init__(self, insertion: Insertion_methods, removal: Removal_methods, matrix, mask, y_train):
        self.insertion_strategy = insertion
        self.removal_strategy = removal
        self.nb = MultinomialNB()   # um modelo por instância (antes era uma variável global)

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

    def _escolhas(self, individual, matrix):
        """Sorteia o rótulo só das linhas DC (as CC são unânimes e já estão em matrix_base)."""
        linhas_dc = matrix[self.dc_indexes]
        return np.array([self.determinator(row, individual[:len(row)]) for row in linhas_dc], dtype=int)

    def fitness_function(self, individual, X_train_transformed, matrix, a, b, g):
        choices = self._escolhas(individual, matrix)
        matrix_2 = self.matrix_base.copy()
        matrix_2[self.dc_indexes] = choices

        self.nb.fit(X_train_transformed, matrix_2)
        df_ = self.nb.predict_log_proba(X_train_transformed)

        dc_sum = np.sum(df_[self.dc_indexes, choices])
        cc_sum = np.sum(df_[self.cc_indexes, self.cc_values])

        cont = np.bincount(choices)
        variance = np.var(cont)

        fitness = a*cc_sum + b*dc_sum - g*variance
        return fitness

    def large_neigborhood_search(self, individual, n_remove, X_train_transformed, matrix, generations,
                                 labels_trans, a, b, g, stop_limit, T_s, T_f):
        best_fitness = float('-inf')
        best_individual = []
        fitness = 0
        cont = 0

        for generation in range(generations):
            indexes = self.removal_strategy.removal(individual, n_remove)
            individual = self.insertion_strategy.insertion(individual, indexes, labels_trans)
            fitness = self.fitness_function(individual, X_train_transformed, matrix, a, b, g)
            mi = (T_f/T_s)**(1/(generations-1))
            T_moment = T_s*mi**generation

            if fitness > best_fitness:
                best_individual = individual.copy()
                best_fitness = fitness
                cont = 0
            else:
                cont += 1
                delta = fitness - best_fitness
                if math.exp(delta/T_moment) > random.random():
                    best_individual = individual
                    best_fitness = fitness

            if cont == stop_limit:
                break

        return best_fitness, best_individual

    def rotular(self, best_individual, matrix):
        """Rótulo final de cada exemplo de treino (o 'choices' que o script original calculava no fim)."""
        rotulos = self.matrix_base.copy()
        rotulos[self.dc_indexes] = self._escolhas(best_individual, matrix)
        return rotulos


# ----------------------------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------------------------
PERCENTAGES = [1, 5, 10, 15, 20]

data_names = [
    # "bbc_text_cls_adapted_dependent_transformed_",
    # "cyberbullying_tweets_adapted_dependent_transformed_",
    # "fake_news_train_adapted_dependent_transformed_",
    # "data_for_AI_adapted_dependent_transformed_", #Novo
    # # "humor_classification_dataset_adapted_dependent_transformed_", #
    # "Emotion_classify_Data_adapted_dependent_transformed_", #Novo
    # # "News_Category_Dataset_v3_adapted_dependent_transformed_", #
    "SMSSpamCollection_adapted_dependent_transformed_",
    #"train_SST2_dependent_transformed_",#
    # "Emotion_classification_adapted_dependent_transformed_",
    # # "Text_classifi1cation_documents_df_file_adapted_dependent_transformed_", #Novo
    # # "train_yelp_dependent_transformed_", #
    # "IMDB_Dataset_adapted_dependent_transformed_",
    # "Womens_Clothing_E-Commerce_Reviews_adapted_dependent_transformed_"
]

# Mesmos métodos do script original (inserção x remoção); cada trial avalia as 4 combinações e usa a média.
INSERCOES = ("moda", "aleatoria")
REMOCOES = ("sequence", "proximidade")

# a, b, g agora são escolhidos pelo Optuna (antes eram fixos: a=5, b=1, g=0.5)
LNS_CFG = dict(n_remove=3, geracoes=2000, stop_limit=30, T_s=2, T_f=0.1)

METRICA = "f1_macro"     # "f1_macro" ou "acuracia"
N_TRIALS = 60            # trials concluídos por estudo
REPETICOES = 1           # execuções do LNS por combinação em cada trial (mais = menos ruído, mais lento)
DB_PATH = "optuna_metalns_big.db"


# ----------------------------------------------------------------------------------------------
# Avaliação de UMA configuração (roda dentro de um processo worker)
# ----------------------------------------------------------------------------------------------
_CACHE = {}   # por processo: cada worker lê/vetoriza cada base uma única vez


def _carregar(nome, porcentagem):
    chave = (nome, porcentagem)
    if chave not in _CACHE:
        path = f"data_base/new_data_bases/{nome}{porcentagem}.csv"
        (_, X_train_transformed, y_train, _, _,
         matrix, labels_trans, _, mask_cc) = data_processor(path, False)
        y_true, matrix, labels_trans = normalizar_rotulos(y_train, matrix, labels_trans)
        _CACHE[chave] = (X_train_transformed, y_train, y_true, matrix, labels_trans, mask_cc)
    return _CACHE[chave]


def avaliar_configuracao(nome, porcentagem, a, b, g, semente, cfg):
    t0 = time.time()
    X_train, y_train, y_true, matrix, labels_trans, mask_cc = _carregar(nome, porcentagem)

    scores = {}
    for ins_nome in INSERCOES:
        for rem_nome in REMOCOES:
            valores = []
            for rep in range(cfg["repeticoes"]):
                random.seed(f"{semente}|{ins_nome}|{rem_nome}|{rep}")   # trial reproduzível

                individual = np.array([random.random() for _ in range(matrix.shape[1])])
                if ins_nome == "moda":
                    insertion = ModaInsertion_big(columns=individual)
                else:
                    insertion = RandomInsertion_big()
                if rem_nome == "sequence":
                    removal = SequenceRemoval()
                else:
                    removal = ClientProximityRemoval_big()

                LNS = MetaLNS_small(insertion=insertion, removal=removal, matrix=matrix,
                                    mask=mask_cc, y_train=y_train)
                _, best_individual = LNS.large_neigborhood_search(
                    individual, cfg["n_remove"], X_train, matrix, cfg["geracoes"], labels_trans,
                    a, b, g, cfg["stop_limit"], cfg["T_s"], cfg["T_f"])
                if len(best_individual) == 0:
                    best_individual = individual

                y_pred = LNS.rotular(best_individual, matrix)
                valores.append(calcular_metrica(y_true, y_pred, cfg["metrica"]))
            scores[f"{ins_nome}+{rem_nome}"] = float(np.mean(valores))

    return {"score": float(np.mean(list(scores.values()))),
            "detalhes": scores,
            "tempo_s": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="MetaLNS_big + Optuna em paralelo")
    ap.add_argument("--trials", type=int, default=N_TRIALS, help="trials concluídos desejados por estudo")
    ap.add_argument("--workers", type=int, default=None,
                    help="processos em paralelo (padrão 3; teto de segurança 4 para 8 GB de RAM)")
    ap.add_argument("--repeticoes", type=int, default=REPETICOES)
    ap.add_argument("--metrica", choices=["f1_macro", "acuracia"], default=METRICA)
    ap.add_argument("--por-infiltracao", action="store_true",
                    help="um estudo separado (mesmo nº de trials) para cada infiltração")
    ap.add_argument("--novo", action="store_true", help="apaga o estudo salvo e recomeça")
    args = ap.parse_args()

    n_workers = workers_seguros(args.workers)
    cfg = {**LNS_CFG, "repeticoes": args.repeticoes, "metrica": args.metrica}

    for nome in data_names:
        grupos = [[p] for p in PERCENTAGES] if args.por_infiltracao else [PERCENTAGES]
        estudos = []
        for porcentagens in grupos:
            sufixo = f"__p{porcentagens[0]}" if len(porcentagens) == 1 else ""
            estudos.append(rodar_estudo(
                nome_estudo=f"metalns_big__{nome}{sufixo}",
                avaliar=avaliar_configuracao,
                kwargs_fixos={"nome": nome, "cfg": cfg},
                porcentagens=porcentagens,
                n_trials=args.trials,
                n_workers=n_workers,
                db_path=DB_PATH,
                config=cfg,
                novo=args.novo,
            ))
        relatorio(estudos, titulo=nome, metrica=args.metrica,
                  csv_path=f"resultados_optuna_metalns_big__{nome}.csv")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Execucao interrompida. Rode o mesmo comando para continuar de onde parou.")