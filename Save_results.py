from sklearn.metrics import (
    precision_score, 
    recall_score,
    f1_score,
    accuracy_score
)
import pandas as pd
from pathlib import Path
import os
from datetime import datetime

def Save_results(type,sub_type,data_name,y_true,y_pred,methods,a,b,g):
    path_ = "Results_Metrics_Project_big.csv"

    if type == False:
        if sub_type == False:
            type_name = "MetaLNS_small"
        else:
            type_name = "MetaLNS_big"
    else:
        if sub_type == False:
            type_name = "MetaGA_small"
        else:
            type_name = "MetaGA_big"

    if len(set(y_true)) == 2:
        new_row = {
            "Data" : data_name,
            "Type" : type_name,
            "Acurracy" : accuracy_score(y_true, y_pred),
            "Precision" : (precision_score(y_true, y_pred, average='binary')),
            "Recall" : (recall_score(y_true, y_pred, average='binary')),
            "F1" : (f1_score(y_true, y_pred, average='binary')),
            "Methods" : methods,
            "a" : a,
            "b" : b,
            "g" : g,
        }

        df_new = pd.DataFrame([new_row])
        
        file_exists = os.path.exists(path_)
        
        df_new.to_csv(path_, mode='a', header=not file_exists, index=False)
    else:
        new_row = {
            "Data" : data_name,
            "Type" : type_name,
            "Acurracy" : accuracy_score(y_true, y_pred),
            "Precision" : (precision_score(y_true, y_pred, average='weighted')),
            "Recall" : (recall_score(y_true, y_pred, average='weighted')),
            "F1" : (f1_score(y_true, y_pred, average='weighted')),
            "Methods" : methods,
            "a" : a,
            "b" : b,
            "g" : g
        }

        df_new = pd.DataFrame([new_row])
        
        file_exists = os.path.exists(path_)
        
        df_new.to_csv(path_, mode='a', header=not file_exists, index=False)
    return()

RESULTS_DIR = Path("Resultados_combinacao")

def salvar_resultados(
    nome_base,
    y_true,
    predictions,
    tempos
):

    pasta_base = RESULTS_DIR / nome_base
    pasta_base.mkdir(parents=True, exist_ok=True)

    metodos = list(predictions.keys())

    resultados_metricas = []

    for metodo in metodos:

        pred = predictions[metodo]

        resultados_metricas.append({
            "base": nome_base,
            "metodo": metodo,
            "acuracia": accuracy_score(y_true, pred),
            "precisao_macro": precision_score(
                y_true, pred,
                average="macro",
                zero_division=0
            ),
            "recall_macro": recall_score(
                y_true, pred,
                average="macro",
                zero_division=0
            ),
            "f1_macro": f1_score(
                y_true, pred,
                average="macro",
                zero_division=0
            ),
            "tempo_segundos": tempos[metodo],
            "data_execucao": datetime.now().isoformat()
        })

    arquivo_metricas = pasta_base / "metricas.csv"

    df_metricas = pd.DataFrame(resultados_metricas)

    df_metricas.to_csv(
        arquivo_metricas,
        index=False,
        encoding="utf-8-sig"
    )

    df_predicoes = pd.DataFrame({
        "id": range(len(y_true)),
        "rotulo_verdadeiro": y_true
    })

    for metodo in metodos:
        df_predicoes[metodo] = predictions[metodo]

    arquivo_predicoes = pasta_base / "predicoes.csv"

    df_predicoes.to_csv(
        arquivo_predicoes,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nResultados salvos para: {nome_base}")
    print(f"Métricas: {arquivo_metricas}")
    print(f"Predições: {arquivo_predicoes}")
