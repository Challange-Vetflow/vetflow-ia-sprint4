"""Treino, comparação e seleção do modelo preditivo de risco de abandono do cuidado.

Compara quatro famílias de modelos (regressão logística, random forest, gradient
boosting e rede neural MLP vista em aula), avalia com validação cruzada e em um
conjunto de teste separado, escolhe o melhor com uma regra explícita e salva tudo
(modelo + metadados + relatório + figuras).

Uso:  python -m vetflow_ai.train
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .prep import MODEL_COLUMNS, checar_qualidade, preparar

MODEL_VERSION = "1.0.0"
# Se o melhor modelo ganhar por menos que isto em AUC (CV), preferimos o mais simples/interpretável.
MARGEM_PREFERENCIA_SIMPLES = 0.01


def construir_pipeline(estimador) -> Pipeline:
    """Imputação (mediana) + padronização para numéricas; moda + one-hot para categóricas."""
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), config.NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), config.CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", estimador)])


def candidatos() -> dict[str, Pipeline]:
    s = config.RANDOM_STATE
    return {
        "Regressão logística": construir_pipeline(LogisticRegression(max_iter=1000, random_state=s)),
        "Random forest": construir_pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=s, n_jobs=-1)),
        "Gradient boosting": construir_pipeline(GradientBoostingClassifier(random_state=s)),
        "Rede neural (MLP)": construir_pipeline(MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
                                                              alpha=1e-3, max_iter=800, early_stopping=True, random_state=s)),
    }


def _metricas(y, proba, limiar=0.5) -> dict:
    pred = (proba >= limiar).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "precisao": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "brier": float(brier_score_loss(y, proba)),
    }


def carregar_dados() -> tuple[pd.DataFrame, pd.Series, dict]:
    bruto = pd.read_csv(config.DATA_PATH)
    df, rel = preparar(bruto)
    problemas = checar_qualidade(df)
    if problemas:
        raise ValueError(f"Dados reprovados na validação: {problemas}")
    rel["problemas_qualidade"] = problemas
    return df[MODEL_COLUMNS], df[config.TARGET], rel


def main() -> dict:
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y, rel_prep = carregar_dados()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    resultados, ajustados, probas = {}, {}, {}
    for nome, pipe in candidatos().items():
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        resultados[nome] = {"cv_auc_media": float(cv_auc.mean()), "cv_auc_desvio": float(cv_auc.std()), **_metricas(y_te, proba)}
        ajustados[nome], probas[nome] = pipe, proba
        print(f"{nome:22s} CV AUC {cv_auc.mean():.3f}±{cv_auc.std():.3f} | teste AUC {resultados[nome]['roc_auc']:.3f}")

    # --- seleção: maior AUC em validação cruzada; empate técnico -> regressão logística (interpretável)
    melhor = max(resultados, key=lambda n: resultados[n]["cv_auc_media"])
    simples = "Regressão logística"
    if melhor == simples:
        escolhido, motivo = simples, "maior AUC em validação cruzada, além de ser o mais simples e interpretável"
    elif resultados[melhor]["cv_auc_media"] - resultados[simples]["cv_auc_media"] < MARGEM_PREFERENCIA_SIMPLES:
        escolhido, motivo = simples, (f"empate técnico (diferença de AUC < {MARGEM_PREFERENCIA_SIMPLES}) com '{melhor}'; "
                                      "preferimos o modelo mais simples e interpretável")
    else:
        escolhido, motivo = melhor, "maior AUC em validação cruzada"
    print(f"\nModelo escolhido: {escolhido} — {motivo}")

    # --- desempenho por nível de alerta (como os limiares se comportam)
    proba_esc = probas[escolhido]
    niveis = pd.cut(proba_esc, [-0.01, config.THRESHOLD_ATENCAO, config.THRESHOLD_CRITICO, 1.01],
                    labels=["NORMAL", "ATENÇÃO", "CRÍTICO"])
    por_nivel = (pd.DataFrame({"nivel": niveis, "y": y_te.values}).groupby("nivel", observed=False)
                 .agg(pets=("y", "size"), taxa_real_risco=("y", "mean")).round(3))
    por_nivel["pets"] = por_nivel["pets"].astype(int)

    # --- importância por permutação (independe do tipo de modelo)
    imp = permutation_importance(ajustados[escolhido], X_te, y_te, scoring="roc_auc", n_repeats=15,
                                 random_state=config.RANDOM_STATE, n_jobs=-1)
    importancias = (pd.Series(imp.importances_mean, index=MODEL_COLUMNS).sort_values(ascending=False))

    # --- retreina o escolhido com TODOS os dados para publicar
    final = candidatos()[escolhido]
    final.fit(X, y)
    baseline = X[config.NUMERIC].median().to_dict()
    meta = {
        "versao": MODEL_VERSION, "modelo": escolhido, "motivo_escolha": motivo,
        "treinado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_treino_total": int(len(X)), "prevalencia_risco": float(y.mean()),
        "limiares": {"atencao": config.THRESHOLD_ATENCAO, "critico": config.THRESHOLD_CRITICO},
        "features": MODEL_COLUMNS, "mediana_features": baseline,
        "dados_sinteticos": True,
    }
    joblib.dump({"pipeline": final, "meta": meta}, config.MODEL_PATH)

    metricas = {"modelos": resultados, "escolhido": escolhido, "motivo": motivo,
                "por_nivel": por_nivel.reset_index().to_dict(orient="records"),
                "recall_por_limiar": {str(t): float(recall_score(y_te, (proba_esc >= t).astype(int))) for t in (0.4, 0.5, 0.7)},
                "importancia_permutacao": importancias.round(4).to_dict(), "preparacao": rel_prep}
    (config.REPORTS_DIR / "metricas.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    _figuras(y_te, probas, escolhido, importancias, por_nivel)
    _relatorio(metricas, por_nivel, importancias, y, y_te)
    print(f"\nModelo salvo em {config.MODEL_PATH}")
    return metricas


def _figuras(y_te, probas, escolhido, importancias, por_nivel) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for nome, p in probas.items():
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, lw=2.4 if nome == escolhido else 1.4, label=f"{nome} (AUC {roc_auc_score(y_te, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="Falsos positivos", ylabel="Verdadeiros positivos", title="Curvas ROC — conjunto de teste")
    ax.legend(loc="lower right", fontsize=8); fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "roc_modelos.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    importancias.sort_values().plot.barh(ax=ax, color="#028090")
    ax.set(xlabel="Queda do AUC ao embaralhar a variável", title=f"O que mais pesa no risco — {escolhido}")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "importancia_variaveis.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    cores = {"NORMAL": "#2a9d8f", "ATENÇÃO": "#e9c46a", "CRÍTICO": "#e76f51"}
    ax.bar(por_nivel.index.astype(str), por_nivel["taxa_real_risco"], color=[cores[n] for n in por_nivel.index.astype(str)])
    for i, (n, row) in enumerate(por_nivel.iterrows()):
        ax.text(i, row["taxa_real_risco"] + 0.01, f"{row['taxa_real_risco']:.0%}\n(n={row['pets']})", ha="center", fontsize=9)
    ax.set(ylim=(0, 1.05), ylabel="Pets que realmente abandonaram", title="Taxa real de risco por nível de alerta")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "risco_por_nivel.png", dpi=150); plt.close(fig)


def _relatorio(m, por_nivel, importancias, y, y_te) -> None:
    linhas = ["# Relatório de avaliação dos modelos — VetFlow\n",
              "> **Atenção:** os dados são **sintéticos** (gerados em `vetflow_ai/data.py`). As métricas abaixo provam que a "
              "arquitetura funciona de ponta a ponta, **não** o desempenho que o modelo teria em clínicas reais.\n",
              f"- Pets no dataset: {len(y)} · prevalência de risco: {y.mean():.1%} · teste: {len(y_te)} pets (25%, estratificado)",
              f"- Preparação: {json.dumps(m['preparacao'], ensure_ascii=False)}\n",
              "## Comparação de modelos\n",
              "| Modelo | AUC (CV 5-fold) | AUC teste | PR-AUC | Precisão | Recall | F1 | Brier |", "|---|---|---|---|---|---|---|---|"]
    for nome, r in m["modelos"].items():
        marca = " ✅" if nome == m["escolhido"] else ""
        linhas.append(f"| {nome}{marca} | {r['cv_auc_media']:.3f} ± {r['cv_auc_desvio']:.3f} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | "
                      f"{r['precisao']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['brier']:.3f} |")
    linhas += [f"\n**Modelo escolhido:** {m['escolhido']} — {m['motivo']}.\n",
               "![ROC](../docs/figures/roc_modelos.png)\n",
               "## Níveis de alerta (limiares 0,40 / 0,70)\n", "| Nível | Pets (teste) | Taxa real de risco |", "|---|---|---|"]
    for nivel, r in por_nivel.iterrows():
        linhas.append(f"| {nivel} | {int(r['pets'])} | {r['taxa_real_risco']:.1%} |")
    rc = m["recall_por_limiar"]
    linhas.append(f"\nCom o corte padrão de 0,5 o modelo recupera {rc['0.5']:.0%} dos pets que de fato abandonaram; usando os cortes do "
                  f"sistema, alertar a partir de ATENÇÃO (≥ 0,40) recupera {rc['0.4']:.0%} e a partir de CRÍTICO (≥ 0,70) {rc['0.7']:.0%}. "
                  "Por isso o fluxo de alertas trata ATENÇÃO como aviso preventivo e CRÍTICO como contato prioritário.")
    linhas += ["\n![Risco por nível](../docs/figures/risco_por_nivel.png)\n",
               "## O que mais pesa no risco (importância por permutação)\n", "| Variável | Queda de AUC |", "|---|---|"]
    for k, v in importancias.items():
        linhas.append(f"| {k} | {v:.4f} |")
    linhas += ["\n![Importância](../docs/figures/importancia_variaveis.png)\n",
               "## Limitações\n",
               "- Rótulo simulado: em produção virá do histórico real (pet que não cumpriu vacina/retorno/dose em até 30 dias).",
               "- O rótulo sintético é gerado por um processo latente quase **linear** (soma ponderada de sinais + ruído). Isso favorece a "
               "regressão logística; com dados reais, com interações mais complexas, random forest, boosting ou MLP podem superá-la. "
               "Por isso o pipeline compara os quatro e a escolha deve ser refeita quando houver dados reais.",
               "- Os quatro modelos ficaram a menos de 0,03 de AUC entre si: a diferença entre eles não é conclusiva com este dataset.",
               "- Nenhum dado de tutor real foi usado; antes de usar em clínica é preciso revalidar com dados reais e com veterinários.",
               "- O score **apoia** a priorização; a decisão clínica continua sendo do veterinário."]
    (config.REPORTS_DIR / "avaliacao_modelos.md").write_text("\n".join(linhas), encoding="utf-8")


if __name__ == "__main__":
    main()
