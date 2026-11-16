"""
Optuna em paralelo para MetaLNS_small / MetaLNS_big
(pensado para notebook com 8 GB de RAM e Windows).

Como funciona
-------------
* O estudo Optuna fica no processo principal (SQLite com um único escritor,
  então não há "database is locked").
* study.optimize(n_jobs=N) cria N threads. Cada thread só despacha a avaliação
  de uma configuração (a, b, g, infiltração) para um pool de N PROCESSOS e
  espera o resultado.
* Assim o trabalho pesado (Python puro + sklearn) roda em N núcleos de verdade,
  sem a trava do GIL, que deixaria o n_jobs "puro" do Optuna (threads) quase
  sem ganho nesse código.
"""
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState
from sklearn.metrics import accuracy_score, f1_score

# Cada worker é um processo Python com pandas/sklearn (~300 MB) + os dados.
# Com 8 GB de RAM (Windows já usa uns 4 GB), 3 é o padrão seguro e 4 o teto.
WORKERS_PADRAO = 3
WORKERS_MAXIMO = 4

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


def workers_seguros(pedido=None):
    """Nº de workers: padrão 3; nunca acima de WORKERS_MAXIMO nem de (threads da CPU - 2)."""
    cpus = os.cpu_count() or 2
    teto = max(1, min(WORKERS_MAXIMO, cpus - 2))
    desejado = WORKERS_PADRAO if pedido is None else pedido
    n = max(1, min(desejado, teto))
    if pedido is not None and n != pedido:
        print(f"[aviso] --workers {pedido} limitado a {n} "
              f"(teto de seguranca para 8 GB de RAM / {cpus} threads de CPU).")
    return n


def calcular_metrica(y_true, y_pred, nome):
    """Qualidade dos rótulos inferidos contra os rótulos reais."""
    if nome == "acuracia":
        return float(accuracy_score(y_true, y_pred))
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def normalizar_rotulos(y_train, matrix, labels):
    """
    Garante rótulos INTEIROS em y_train, matrix e labels, seja qual for a versão do Data_processor
    (com pandas 3 / Python 3.14 os rótulos podem chegar como texto: 'ham'/'spam').
    Codificação alfabética (ham=0, spam=1), a mesma do LabelEncoder.
    Devolve (y_true, matrix, labels) já convertidos.
    """
    y = np.asarray(y_train).ravel()
    lab = np.asarray(labels).ravel()
    m = np.asarray(matrix)

    def achatar(arr):
        for item in arr.ravel():
            if isinstance(item, (list, tuple, np.ndarray)):
                yield from item
            else:
                yield item

    vistos = set(y.tolist()) | set(lab.tolist()) | set(achatar(m))
    textos = sorted(v for v in vistos if isinstance(v, str))
    if not textos:
        return y, matrix, np.unique(lab)                      # já estava tudo em inteiros

    mapa = {v: i for i, v in enumerate(textos)}                # alfabético, igual ao LabelEncoder

    def cv(v):                                                 # texto -> inteiro; inteiro fica como está
        return mapa[v] if isinstance(v, str) else int(v)

    y_i = np.array([cv(v) for v in y.tolist()], dtype=int)
    lab_i = np.array(sorted({cv(v) for v in lab.tolist()}), dtype=int)
    if m.ndim == 2:                                            # matriz 2-D (small)
        m_i = np.array([[cv(v) for v in linha] for linha in m.tolist()], dtype=int)
    else:                                                      # vetor de inteiros/listas (big)
        m_i = np.empty(len(m), dtype=object)
        for i, item in enumerate(m):
            m_i[i] = ([cv(v) for v in item] if isinstance(item, (list, tuple, np.ndarray))
                      else cv(item))
    return y_i, m_i, lab_i


def rodar_estudo(nome_estudo, avaliar, kwargs_fixos, porcentagens, n_trials, n_workers,
                 db_path, config, seed=42, novo=False):
    """
    Cria (ou retoma) um estudo Optuna e roda os trials restantes em paralelo.

    avaliar        : função de nível de módulo (precisa ser "picklável" para o Windows/spawn);
                     recebe porcentagem, a, b, g, semente + kwargs_fixos e devolve
                     {"score": float, "detalhes": dict, "tempo_s": float}.
    porcentagens   : lista com >1 item -> a infiltração vira parâmetro categórico do estudo;
                     lista com 1 item  -> infiltração fixa (modo --por-infiltracao).
    n_trials       : total de trials CONCLUÍDOS desejado no estudo (ao retomar, roda só o que falta).
    config         : dict com tudo que muda o significado do score; se mudar entre execuções,
                     o estudo salvo é recusado (use --novo) para não misturar resultados.
    """
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": 60}},   # espera o lock do SQLite em vez de falhar
    )
    if novo:
        try:
            optuna.delete_study(study_name=nome_estudo, storage=storage)
        except KeyError:
            pass

    study = optuna.create_study(
        study_name=nome_estudo,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        # constant_liar: evita sugerir pontos parecidos enquanto outros trials ainda estão rodando
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, constant_liar=True),
    )

    anterior = study.user_attrs.get("config")
    if anterior is not None and anterior != config:
        raise SystemExit(
            f"[{nome_estudo}] a configuracao mudou desde a ultima execucao deste estudo.\n"
            f"  antes : {anterior}\n  agora : {config}\n"
            f"Rode com --novo para recomecar (ou apague {db_path})."
        )
    study.set_user_attr("config", config)

    # Trials "RUNNING" que sobraram de uma execução interrompida (Ctrl+C, queda de energia...)
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(t.number, state=TrialState.FAIL)

    feitos = len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)))
    restantes = n_trials - feitos
    if restantes <= 0:
        print(f"[{nome_estudo}] ja tem {feitos} trials concluidos (>= {n_trials}); nada a rodar.")
        return study

    print(f"[{nome_estudo}] {restantes} trials a rodar com {n_workers} worker(s) "
          f"({feitos} ja concluidos).")

    with ProcessPoolExecutor(max_workers=n_workers) as pool:

        def objective(trial):
            if len(porcentagens) > 1:
                p = trial.suggest_categorical("porcentagem", porcentagens)
            else:
                p = porcentagens[0]
            a = trial.suggest_int("a", 0, 10)
            b = trial.suggest_int("b", 0, 10)
            g = trial.suggest_int("g", 0, 10)
            trial.set_user_attr("porcentagem", p)

            futuro = pool.submit(avaliar, porcentagem=p, a=a, b=b, g=g,
                                 semente=trial.number, **kwargs_fixos)
            try:
                resultado = futuro.result()
            except BrokenProcessPool as erro:
                raise RuntimeError(
                    "Um processo worker foi encerrado (provavel falta de RAM). "
                    "Rode de novo com menos --workers."
                ) from erro

            trial.set_user_attr("detalhes", resultado["detalhes"])
            trial.set_user_attr("tempo_s", resultado["tempo_s"])
            return resultado["score"]

        try:
            study.optimize(objective, n_trials=restantes, n_jobs=n_workers)
        except KeyboardInterrupt:
            print("\nInterrompido (Ctrl+C). Os trials ja concluidos ficaram salvos em", db_path)
            pool.shutdown(wait=False, cancel_futures=True)
            raise   # encerra o programa inteiro (nao segue para o proximo estudo)

    return study


def relatorio(estudos, titulo, metrica, csv_path=None):
    """Imprime a melhor configuração global e a melhor de cada infiltração; salva os trials em CSV."""
    trials = [t for e in estudos for t in e.trials if t.state == TrialState.COMPLETE]
    linha = "=" * 72
    print(f"\n{linha}\n{titulo}   |   metrica: {metrica}\n{linha}")
    if not trials:
        print("Nenhum trial concluido.")
        return

    melhor = max(trials, key=lambda t: t.value)
    print(f"Melhor configuracao (trial #{melhor.number}): "
          f"infiltracao={melhor.user_attrs.get('porcentagem')}%  "
          f"a={melhor.params['a']}  b={melhor.params['b']}  g={melhor.params['g']}  "
          f"->  {metrica}={melhor.value:.4f}")

    print("\nMelhor a, b, g de cada infiltracao:")
    print(f"  {'infiltr.':>8}  {metrica:>10}   a   b   g   trials")
    for p in sorted({t.user_attrs.get("porcentagem") for t in trials}):
        do_p = [t for t in trials if t.user_attrs.get("porcentagem") == p]
        m = max(do_p, key=lambda t: t.value)
        print(f"  {str(p) + '%':>8}  {m.value:>10.4f}  {m.params['a']:>2}  {m.params['b']:>2}  "
              f"{m.params['g']:>2}   {len(do_p)}")
    print("\nObs.: a infiltracao muda a dificuldade da base (menos infiltracao tende a dar score maior);\n"
          "      para escolher a, b, g compare dentro de cada infiltracao.")

    if csv_path:
        df = pd.concat([e.trials_dataframe().assign(estudo=e.study_name) for e in estudos],
                       ignore_index=True)
        # ';' e ',' decimal + utf-8-sig: abre direto no Excel em portugues
        df.to_csv(csv_path, index=False, sep=";", decimal=",", encoding="utf-8-sig")
        print(f"\nTodos os trials salvos em: {csv_path}")