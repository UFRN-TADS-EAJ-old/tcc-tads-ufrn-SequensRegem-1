import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def data_processor(path,big_or_small,AG_or_LNS):
    data = pd.read_csv(filepath_or_buffer = path)

    labels = data["labels"].unique()
    data['text'] = data['text'].fillna('').astype(str)
    
    X = data["text"]
    y = data["labels"]
    labels = y.unique()
    
    if y.dtype == object or str:
        flag = True
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(y)
        y = pd.DataFrame(y)
    
    columns = data.iloc[:,2:]

    nunique_row = columns.nunique(axis=1)

    mask_cc = nunique_row == 1

    df_cc = data[mask_cc]
    df_dc = data[~mask_cc]
    
    if df_cc.empty == False:
        y_cc = df_cc['labels']
        X_cc = df_cc['text']
        y_dc = df_dc['labels']
        X_dc = df_dc['text']
    else:
        y_dc = df_dc['labels']
        X_dc = df_dc['text']
    
    if df_cc.empty == True:
        X_train, X_test, y_train, y_test = train_test_split(X_dc, y_dc, test_size = 0.33, random_state = 1, stratify = y)
    else:
        X_train_cc, X_test_cc, y_train_cc, y_test_cc = train_test_split(X_cc, y_cc, test_size = 0.35, random_state = 1, stratify = y_cc)
        X_train_dc, X_test_dc, y_train_dc, y_test_dc = train_test_split(X_dc, y_dc, test_size = 0.65, random_state = 1, stratify = y_dc)
        
        X_train = pd.concat([X_train_cc,X_train_dc], ignore_index=True)
        X_test = pd.concat([X_test_cc,X_test_dc], ignore_index=True)
        y_train = pd.concat([y_train_cc,y_train_dc], ignore_index=True)
        y_test = pd.concat([y_test_cc,y_test_dc], ignore_index=True)
    
    vectorizer = TfidfVectorizer(stop_words='english')

    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)
    
    indices_treino = y_train.index
    columns = data.iloc[indices_treino, 2:]

    columns = columns.apply(le.transform)
    # if columns.dtypes.iloc[0] == 'object' or str(columns.dtypes.iloc[0]) == 'string':
    #     for col in columns.columns:
    #         columns[col] = le.transform(columns[col].astype(str))

    matrix = columns.to_numpy()
    
    matrix = np.array(columns)
    
    matrix = np.array([list(dict.fromkeys(lista)) for lista in matrix], dtype=object)


    if (y_train.dtype == str) or (y_train.dtype == object):
        y_train = le.transform(y_train)

    if AG_or_LNS == True:
        
        if big_or_small == True:
            individual_lenght = 0
            for index,value in enumerate(matrix):
                if (value == value[0]).all() != True:
                    individual_lenght+=1
        else:
            individual_lenght = (len(data.columns)-2)
        
        labels_trans = le.transform(labels)

        if matrix.dtype == str:
            matrix = le.transform(matrix)
        
        return(X_train_transformed,y_train,X_test_transformed,y_test,labels_trans,individual_lenght,matrix,le,flag)
    else:
        if big_or_small == True:

            matrix_ = matrix.copy()

            len_ = 0
            for _ in matrix_:
                if 1 < len(_):
                    len_+=1

            labels_trans = le.transform(labels)

            individual = random.choices(labels_trans, k=len_)

            individual = np.array(individual)

            if matrix.dtype == str:
                matrix = le.transform(matrix)

            return(individual,X_train_transformed,y_train,X_test_transformed,y_test,matrix,labels_trans,le)
        else:

            individual_lenght = (len(data.columns)-2)

            individual = np.array([])

            for i in range(individual_lenght):
                individual = np.append(individual,random.random())

            labels_trans = le.transform(labels)

            if matrix.dtype == str:
                matrix = le.transform(matrix)

            return(individual,X_train_transformed,y_train,X_test_transformed,y_test,matrix,labels_trans,le)