from sklearn.metrics import (
    precision_score, 
    recall_score,
    f1_score,
    accuracy_score
)
import pandas as pd
import os

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