"""
Snorkel Label Model + Stacking otimizados com Optuna, 30 repetições por configuração, em paralelo.

Rode pelo TERMINAL, na raiz do projeto (não pelo Spyder/Jupyter, que travam o multiprocessing no Windows):

    python Combinations_Optuna.py
    python Combinations_Optuna.py --workers 4 --repeticoes 30
    python Combinations_Optuna.py --novo          # apaga estudos e resultados salvos e recomeça

Como funciona
-------------
Para cada base, cada infiltração (1, 5, 10, 15, 20) e cada método (Snorkel, Stacking) é criado um estudo
Optuna. Um "trial" = uma configuração de hiperparâmetro:

    Snorkel  -> n_epochs in [100, 250, 500, 750]
    Stacking -> max_iter in [500, 700, 1000, 1500]

Cada trial executa a configuração N_REPETICOES (30) vezes na mesma base, cada repetição com uma divisão
treino/teste diferente (random_state = nº da repetição). O valor que o Optuna maximiza é a MÉDIA do
F1-score macro nas 30 repetições. As 30 repetições de um trial rodam em paralelo.

Sementes
--------
  - divisão treino/teste: random_state = nº da repetição (1..30), igual para todas as configurações;
  - Snorkel LabelModel: seed = 42 (fixa, como no código original);
  - Stacking (LogisticRegression/lbfgs): determinístico, não usa semente.

Saída: UM CSV POR BASE  ->  Resultados_combinacao_optuna/resultados_<nome_da_base>.csv
-----------------------------------------------------------------------------------
Ex.: resultados_bbc_text_cls_adapted_dependent_transformed.csv, resultados_cyberbullying_tweets_...csv, etc.
Cada CSV tem uma linha por repetição (30 linhas por configuração), com as colunas:
  base, infiltracao, metodo, parametro, valor_parametro, repeticao, random_state_divisao, f1_macro,
  f1_macro_medio_config, f1_macro_desvio_config, n_repeticoes_config, melhor_configuracao
As três últimas colunas resumem a configuração (média/desvio das 30 repetições e se ela é a melhor
daquela base/infiltração/método); são preenchidas ao fim de cada infiltração. O CSV é gravado repetição
a repetição, então a execução pode ser interrompida e retomada de onde parou. Nenhuma predição é salva.
"""
import os

# 1 thread de BLAS por processo: evita vários workers disputando os mesmos núcleos.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from snorkel.labeling.model import LabelModel


# ----------------------------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------------------------
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
    "data_base/new_data_bases/Womens_Clothing_E-Commerce_Reviews_adapted_dependent_transformed_",
]

PERCENTAGES = [1, 5, 10, 15, 20]
N_REPETICOES = 30
TEST_SIZE = 0.33
SNORKEL_SEED = 42
LF_COLUMNS = [f"Labels_{i}" for i in range(10, 20)]

# Espaço de busca de cada método: (nome do parâmetro, valores possíveis)
ESPACO = {
    "Snorkel": ("n_epochs", [100, 250, 500, 750, 1000]),
    "Stacking": ("max_iter", [500, 700, 1000, 1500, 2000]),
}

RESULTS_DIR = Path("Resultados_combinacao_optuna")
OPTUNA_DB = RESULTS_DIR / "optuna_combinacoes.db"



def caminho_csv(nome_base):
    """Um CSV por base: resultados_<nome_da_base>.csv (sem o '_' final do nome)."""
    return RESULTS_DIR / f"resultados_{nome_base.rstrip('_')}.csv"


COLUNAS_REP = ["base", "infiltracao", "metodo", "parametro", "valor_parametro",
               "repeticao", "random_state_divisao", "f1_macro"]
COLUNAS_RESUMO = ["f1_macro_medio_config", "f1_macro_desvio_config",
                  "n_repeticoes_config", "melhor_configuracao"]
COLUNAS_CSV = COLUNAS_REP + COLUNAS_RESUMO
CHAVE_REP = ["base", "infiltracao", "metodo", "valor_parametro", "repeticao"]
CHAVE_CONFIG = ["base", "infiltracao", "metodo", "parametro", "valor_parametro"]


# ----------------------------------------------------------------------------------------------
# Métodos de combinação (com o hiperparâmetro exposto)
# ----------------------------------------------------------------------------------------------
def snorkel_label_model(L_train, L_test, num_classes, n_epochs):
    label_model = LabelModel(cardinality=num_classes, verbose=False)
    label_model.fit(L_train=L_train, n_epochs=n_epochs, log_freq=100, seed=SNORKEL_SEED)
    return label_model.predict(L=L_test, tie_break_policy="abstain")


def stacking(L_train, y_train, L_test, max_iter):
    modelo = make_pipeline(
        OneHotEncoder(handle_unknown="ignore"),
        LogisticRegression(max_iter=max_iter, class_weight="balanced"),
    )
    modelo.fit(L_train, y_train)
    return modelo.predict(L_test)


def f1_macro(y_true, y_pred):
    """
    F1 macro sobre as classes reais. Previsões 'ABSTAIN' (-1 do Snorkel) contam como erro,
    mas não viram uma classe extra (senão derrubariam artificialmente o F1 macro).
    """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    classes = sorted(set(y_true))
    return float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))


# ----------------------------------------------------------------------------------------------
# Execução de UMA repetição (roda dentro de um processo worker; NÃO escreve em disco)
# ----------------------------------------------------------------------------------------------
_CACHE = {}   # por processo: cada worker lê cada CSV de base uma única vez


def _carregar(caminho):
    if caminho not in _CACHE:
        df = pd.read_csv(caminho)
        faltantes = [c for c in ["labels", *LF_COLUMNS] if c not in df.columns]
        if faltantes:
            raise KeyError(f"Colunas ausentes em {caminho}: {faltantes}")
        encoder = LabelEncoder().fit(df["labels"])
        _CACHE[caminho] = (df, encoder)
    return _CACHE[caminho]


def executar_repeticao(base, pct, metodo, valor, repeticao):
    """Roda 1 repetição de 1 configuração e devolve a linha de resultado."""
    warnings.filterwarnings("ignore")

    df, encoder = _carregar(f"{base}{pct}.csv")

    df_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=repeticao, stratify=df["labels"]
    )
    L_train, L_test = df_train[LF_COLUMNS], df_test[LF_COLUMNS]
    y_train, y_test = df_train["labels"], df_test["labels"]

    if metodo == "Snorkel":
        L_train_enc = L_train.apply(lambda c: encoder.transform(c)).to_numpy()
        L_test_enc = L_test.apply(lambda c: encoder.transform(c)).to_numpy()
        pred_cod = snorkel_label_model(
            L_train_enc, L_test_enc, num_classes=len(encoder.classes_), n_epochs=valor
        )
        predicoes = ["ABSTAIN" if p == -1 else encoder.inverse_transform([int(p)])[0] for p in pred_cod]
    else:
        predicoes = list(stacking(L_train, y_train, L_test, max_iter=valor))

    return {
        "base": Path(base).name,
        "infiltracao": pct,
        "metodo": metodo,
        "parametro": ESPACO[metodo][0],
        "valor_parametro": valor,
        "repeticao": repeticao,
        "random_state_divisao": repeticao,
        "f1_macro": f1_macro(y_test, predicoes),
    }


# ----------------------------------------------------------------------------------------------
# CSV único (só o processo principal escreve)
# ----------------------------------------------------------------------------------------------
def ler_csv(nome_base):
    caminho = caminho_csv(nome_base)
    if not caminho.is_file():
        return pd.DataFrame(columns=COLUNAS_CSV)
    df = pd.read_csv(caminho)
    return df.reindex(columns=COLUNAS_CSV)


def anexar_linha(linha):
    """Acrescenta uma repetição ao CSV (cria o arquivo com cabeçalho se não existir)."""
    caminho = caminho_csv(linha["base"])
    existe = caminho.is_file()
    pd.DataFrame([linha]).reindex(columns=COLUNAS_CSV).to_csv(
        caminho, mode="a", header=not existe, index=False, encoding="utf-8-sig"
    )


def remover_estudo_do_csv(nome_base, pct, metodo):
    df = ler_csv(nome_base)
    if df.empty:
        return
    manter = ~((df["base"] == nome_base) & (df["infiltracao"] == pct) & (df["metodo"] == metodo))
    df[manter].to_csv(caminho_csv(nome_base), index=False, encoding="utf-8-sig")


def consolidar(nome_base):
    """Reescreve o CSV da base preenchendo média/desvio/melhor configuração (chamado entre estudos, sem concorrência)."""
    caminho = caminho_csv(nome_base)
    df = ler_csv(nome_base)
    if df.empty:
        return
    df = df.drop_duplicates(subset=CHAVE_REP, keep="last")
    g = df.groupby(CHAVE_CONFIG)["f1_macro"]
    df["f1_macro_medio_config"] = g.transform("mean")
    df["f1_macro_desvio_config"] = g.transform("std")
    df["n_repeticoes_config"] = g.transform("count")

    medias = df.drop_duplicates(subset=CHAVE_CONFIG)[CHAVE_CONFIG + ["f1_macro_medio_config"]]
    idx_melhor = medias.groupby(["base", "infiltracao", "metodo"])["f1_macro_medio_config"].idxmax()
    melhores = medias.loc[idx_melhor, CHAVE_CONFIG]
    df["melhor_configuracao"] = df.merge(
        melhores.assign(_m=True), on=CHAVE_CONFIG, how="left"
    )["_m"].fillna(False).to_numpy()

    df = df.sort_values(CHAVE_REP)
    tmp = caminho.with_suffix(".tmp")
    df[COLUNAS_CSV].to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, caminho)

    print("\n=== Melhor configuração por base / infiltração / método ===")
    resumo = df[df["melhor_configuracao"]].drop_duplicates(subset=CHAVE_CONFIG)
    print(resumo[CHAVE_CONFIG + ["f1_macro_medio_config", "n_repeticoes_config"]].to_string(index=False))


# ----------------------------------------------------------------------------------------------
# Estudo Optuna (processo principal; as repetições vão para o pool)
# ----------------------------------------------------------------------------------------------
def rodar_estudo(pool, base, pct, metodo, n_repeticoes, novo):
    nome_base = Path(base).name
    nome_param, valores = ESPACO[metodo]
    nome_estudo = f"combinacao__{metodo}__{nome_base}__p{pct}"
    storage = f"sqlite:///{OPTUNA_DB.as_posix()}"

    if novo:
        try:
            optuna.delete_study(study_name=nome_estudo, storage=storage)
        except KeyError:
            pass
        remover_estudo_do_csv(nome_base, pct, metodo)

    study = optuna.create_study(
        study_name=nome_estudo,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.GridSampler({nome_param: valores}),
    )

    def objective(trial):
        valor = trial.suggest_categorical(nome_param, valores)
        t0 = time.perf_counter()

        # Repetições já gravadas no CSV (retomada) não são refeitas
        df = ler_csv(nome_base)
        sel = df[(df["base"] == nome_base) & (df["infiltracao"] == pct)
                 & (df["metodo"] == metodo) & (df["valor_parametro"] == valor)]
        scores = {int(r): float(s) for r, s in zip(sel["repeticao"], sel["f1_macro"])}
        faltam = [r for r in range(1, n_repeticoes + 1) if r not in scores]

        futuros = [pool.submit(executar_repeticao, base, pct, metodo, valor, rep) for rep in faltam]
        for fut in as_completed(futuros):
            linha = fut.result()
            anexar_linha(linha)                       # grava assim que cada repetição termina
            scores[linha["repeticao"]] = linha["f1_macro"]

        arr = np.array([scores[r] for r in range(1, n_repeticoes + 1)])
        trial.set_user_attr("desvio_padrao", float(arr.std(ddof=1)))
        print(f"    {metodo} {nome_param}={valor:<5} F1 macro médio={arr.mean():.4f} "
              f"± {arr.std(ddof=1):.4f}  ({time.perf_counter() - t0:.1f}s)")
        return float(arr.mean())

    concluidos = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pendentes = len(valores) - len(concluidos)
    if pendentes > 0:
        study.optimize(objective, n_trials=pendentes)
    else:
        print(f"    {metodo}: estudo já concluído; reaproveitando resultados.")
    return study


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Snorkel e Stacking + Optuna, 30 repetições, em paralelo")
    ap.add_argument("--workers", type=int, default=None,
                    help="processos em paralelo (padrão: nº de núcleos - 1, no máximo 4)")
    ap.add_argument("--repeticoes", type=int, default=N_REPETICOES)
    ap.add_argument("--novo", action="store_true", help="apaga estudos e resultados salvos e recomeça")
    args = ap.parse_args()

    n_workers = args.workers or max(1, min(4, (os.cpu_count() or 2) - 1))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"Workers: {n_workers} | Repetições por configuração: {args.repeticoes} | Métrica: F1 macro")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for base in bases:
            for pct in PERCENTAGES:
                print(f"\nBase: {Path(base).name} | Infiltração: {pct}%")
                for metodo in ESPACO:
                    rodar_estudo(pool, base, pct, metodo, args.repeticoes, args.novo)
                consolidar(Path(base).name)   # preenche média/desvio/melhor ao fim de cada infiltração

    print("\nResultados salvos em:")
    for base in bases:
        print("  ", caminho_csv(Path(base).name).resolve())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Execução interrompida. Rode o mesmo comando para continuar de onde parou.")