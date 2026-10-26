import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
 
def data_processor(path, big_or_small):
    data = pd.read_csv(filepath_or_buffer = path)

    labels = data["labels"].unique()
    data['text'] = data['text'].fillna('').astype(str)
    
    X = data["text"]
    y = data["labels"]

    columns = data.iloc[:, 2:].to_numpy()
    
    # Inicializa o LabelEncoder de forma robusta para lidar com strings (como 'spam' / 'ham')
    le = LabelEncoder()
    flag = False

    # Verifica se os dados nas colunas ou nos rótulos contêm texto/string
    if columns.dtype == object or y.dtype == object or not np.issubdtype(columns.dtype, np.number):
        flag = True
        # Une todos os valores únicos possíveis para ajustar o encoder sem erros de classes não vistas
        all_unique_vals = np.unique(np.concatenate([y.astype(str).unique(), columns.ravel().astype(str)]))
        le.fit(all_unique_vals)
        
        y = le.transform(y.astype(str))
        y = pd.DataFrame(y)
        
        original_shape = columns.shape
        columns = le.transform(columns.ravel().astype(str)).reshape(original_shape)
    else:
        le.fit(labels)

    nunique_row = [len(np.unique(row)) for row in columns]
    nunique_row = np.asarray(nunique_row)

    mask_cc = nunique_row == 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 1, stratify = y)
    vectorizer = TfidfVectorizer(stop_words='english')

    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)
    
    indices_treino = y_train.index
    train_columns = columns[indices_treino]

    matrix = train_columns.copy()

    if big_or_small == True:
        matrix_ = matrix.copy()

        mask_cleaner = np.hstack([np.ones((matrix_.shape[0], 1), dtype=bool), matrix_[:, 1:] != matrix_[:, :-1]])
        cleaned_matrix = [matrix_[i, mask_cleaner[i]].tolist() for i in range(matrix_.shape[0])]
        cleaned_matrix = np.array([sub[0] if len(sub) == 1 else sub for sub in cleaned_matrix], dtype=object)

        len_ = (~mask_cc[y_train.index]).sum()
        individual = random.choices(list(range(len(le.classes_))) if flag else list(np.unique(matrix)), k=len_)
        individual = np.array(individual)

        return(individual, X_train_transformed, y_train, X_test_transformed, y_test, cleaned_matrix, le.classes_ if flag else labels, le, mask_cc)
    else:
        individual_lenght = (len(data.columns) - 2)
        individual = np.array([])

        for i in range(individual_lenght):
            individual = np.append(individual, random.random())

        return(individual, X_train_transformed, y_train, X_test_transformed, y_test, matrix, le.classes_ if flag else labels, le, mask_cc)