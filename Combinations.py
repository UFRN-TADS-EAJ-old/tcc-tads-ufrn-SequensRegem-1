import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from Save_results import salvar_resultados
from Combination_methods import (
    majority_voting,
    snorkel_label_model,
    stacking
)

bases = [
    "data_base/new_data_bases/bbc_text_cls_adapted_dependent_transformed_",
    "data_base/new_data_bases/cyberbullying_tweets_adapted_dependent_transformed_",
    "data_base/new_data_bases/fake_news_train_adapted_dependent_transformed_",
    "data_base/new_data_bases/data_for_AI_adapted_dependent_transformed_",
    "data_base/new_data_bases/Emotion_classify_Data_adapted_dependent_transformed_",
    "data_base/new_data_bases/SMSSpamCollection_adapted_dependent_transformed_",
    "data_base/new_data_bases/train_SST2_dependent_transformed_",
    "data_base/new_data_bases/Emotion_classification_adapted_dependent_transformed_",
    "data_base/new_data_bases/IMDB_Dataset_adapted_dependent_transformed_",
    "data_base/new_data_bases/Womens_Clothing_E-Commerce_Reviews_adapted_dependent_transformed_"
]

PERCENTAGES = [1, 5, 10, 15, 20]

RESULTS_DIR = Path("Resultados_combinacao")

for base in bases:

    nome_base = Path(base).name

    for percentage in PERCENTAGES:

        caminho = f"{base}{percentage}.csv"

        df = pd.read_csv(filepath_or_buffer=caminho)

        pasta_resultados = (
            RESULTS_DIR / nome_base / f"p{percentage}"
        )

        pasta_resultados.mkdir(
            parents=True,
            exist_ok=True
        )

        df_train, df_test = train_test_split(
            df,
            test_size=0.33,
            random_state=1,
            stratify=df["labels"]
        )

        lf_columns = [
            f"Labels_{i}" for i in range(10, 20)
        ]

        L_train = df_train[lf_columns]
        L_test = df_test[lf_columns]

        y_train = df_train["labels"]
        y_test = df_test["labels"]

        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder()

        encoder.fit(df["labels"])

        y_train_encoded = encoder.transform(y_train)
        y_test_encoded = encoder.transform(y_test)

        L_train_encoded = L_train.apply(
            lambda col: encoder.transform(col)
        ).to_numpy()

        L_test_encoded = L_test.apply(
            lambda col: encoder.transform(col)
        ).to_numpy()

        import time

        tempos = {}
        predictions = {}

        # ---------------------------------
        # 1. MAJORITY VOTING
        # ---------------------------------

        inicio = time.perf_counter()

        pred_mv = majority_voting(L_test_encoded)

        tempos["Majority_Voting"] = time.perf_counter() - inicio


        # ---------------------------------
        # 2. SNORKEL LABEL MODEL
        # ---------------------------------

        inicio = time.perf_counter()

        pred_snorkel = snorkel_label_model(
            L_train_encoded,
            L_test_encoded,
            num_classes=len(encoder.classes_)
        )

        tempos["Snorkel"] = time.perf_counter() - inicio


        # ---------------------------------
        # 3. STACKING
        # ---------------------------------

        inicio = time.perf_counter()

        pred_stacking = stacking(
            L_train,
            y_train,
            L_test
        )

        tempos["Stacking"] = time.perf_counter() - inicio


        # ---------------------------------
        # 4. CONVERTER PREVISÕES
        #    PARA OS RÓTULOS ORIGINAIS
        # ---------------------------------

        def converter_rotulos(predicoes, encoder):

            return [
                "ABSTAIN" if p == -1
                else encoder.inverse_transform([int(p)])[0]
                for p in predicoes
            ]


        predictions["Majority_Voting"] = converter_rotulos(
            pred_mv, encoder
        )

        predictions["Snorkel"] = converter_rotulos(
            pred_snorkel, encoder
        )

        predictions["Stacking"] = list(pred_stacking)


        # ---------------------------------
        # 5. SALVAR RESULTADOS
        # ---------------------------------

        salvar_resultados(
            nome_base,
            y_test,
            predictions,
            tempos
        )