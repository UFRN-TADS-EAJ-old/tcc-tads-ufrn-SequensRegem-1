import numpy as np
from scipy.stats import mode
from snorkel.labeling.model import LabelModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

def majority_voting(L):

    L = np.asarray(L)

    # Verifica se os rótulos são numéricos
    if not np.issubdtype(L.dtype, np.integer):
        raise TypeError(
            "Majority Voting exige rótulos inteiros. "
            "Utilize L_train_encoded ou L_test_encoded."
        )

    resultados = []

    for linha in L:

        # Remove abstenções (-1)
        votos = linha[linha != -1]

        if len(votos) == 0:
            resultados.append(-1)

        else:
            contagens = np.bincount(votos)
            resultados.append(np.argmax(contagens))

    return np.array(resultados)


def snorkel_label_model(L_train, L_test, num_classes):

    label_model = LabelModel(
        cardinality=num_classes,
        verbose=False
    )

    label_model.fit(
        L_train=L_train,
        n_epochs=500,
        log_freq=100,
        seed=42
    )

    predictions = label_model.predict(
        L=L_test,
        tie_break_policy="abstain"
    )

    return predictions

def stacking(L_train, y_train, L_test):

    modelo = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )

    modelo.fit(L_train, y_train)

    predictions = modelo.predict(L_test)

    return predictions