"""Inferência: transforma os dados de um pet em score, nível de alerta, motivos e ação recomendada.

O score vem SEMPRE do modelo preditivo (determinístico). A explicação mostra quanto cada grupo
de fatores (vacinação, tratamento, acompanhamento...) contribui para o risco DAQUELE pet:
para cada grupo, neutralizamos suas variáveis (voltam à mediana da população) e medimos
quanto o risco cai (em log-odds). O `peso` de um grupo é a parcela do risco total daquele pet
que ele explica. É isso que personaliza a ação recomendada.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .prep import MODEL_COLUMNS, RAW_COLUMNS, preparar


def nivel_de_risco(score: float) -> str:
    if score >= config.THRESHOLD_CRITICO:
        return "CRÍTICO"
    if score >= config.THRESHOLD_ATENCAO:
        return "ATENÇÃO"
    return "NORMAL"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _num(v) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _motivo(grupo: str, r: pd.Series) -> str:
    """Frase com FATOS do pet (sem inventar nada) para o tutor/clínica e para o agente de IA."""
    if grupo == "Vacinação":
        z, n = _num(r.get("razao_atraso_vacina")), _num(r.get("vacinas_em_atraso"))
        partes = []
        if z is not None:
            razao_txt = f"{z:.1f}".replace(".", ",")  # pt-BR
            partes.append(f"última vacina com {razao_txt}x o intervalo esperado")
        if n:
            partes.append(f"{int(n)} vacina(s) em atraso")
        return ", ".join(partes) or "vacinação fora do esperado"
    if grupo == "Tratamento":
        a, d = _num(r.get("adesao_tratamento")), _num(r.get("doses_perdidas"))
        partes = []
        if a is not None:
            partes.append(f"adesão ao tratamento de {a:.0%}")
        if d:
            partes.append(f"{int(d)} dose(s) perdida(s)")
        return ", ".join(partes) or "tratamento com baixa adesão"
    if grupo == "Acompanhamento clínico":
        dias, f = _num(r.get("dias_desde_ultima_consulta")), _num(r.get("faltas_consultas"))
        partes = []
        if dias is not None:
            partes.append(f"última consulta há {int(dias)} dias")
        if f:
            partes.append(f"{int(f)} falta(s) em consultas")
        return ", ".join(partes) or "acompanhamento clínico irregular"
    if grupo == "Engajamento no app":
        t, lg = _num(r.get("taxa_lembretes_ignorados")), _num(r.get("dias_desde_ultimo_login_app"))
        partes = []
        if t is not None:
            partes.append(f"{t:.0%} dos lembretes ignorados")
        if lg is not None:
            partes.append(f"sem abrir o app há {int(lg)} dias")
        return ", ".join(partes) or "baixo engajamento com o app"
    if grupo == "Comportamento":
        return f"{int(_num(r.get('alertas_comportamento')) or 0)} alerta(s) de comportamento recentes"
    return grupo


class Predictor:
    def __init__(self, caminho: Path | str = config.MODEL_PATH):
        artefato = joblib.load(caminho)
        self.pipeline = artefato["pipeline"]
        self.meta: dict = artefato["meta"]
        self.baseline: dict = self.meta["mediana_features"]

    # ---------------------------------------------------------------- núcleo
    def _scores(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[MODEL_COLUMNS])[:, 1]

    def _contribuicoes(self, X: pd.DataFrame, base: np.ndarray) -> dict[str, np.ndarray]:
        """Contribuição de cada grupo ao risco, em log-odds (> 0 = o grupo AUMENTA o risco do pet)."""
        out = {}
        for grupo, cols in config.GRUPOS.items():
            Xn = X[MODEL_COLUMNS].copy()
            for c in cols:
                Xn[c] = self.baseline[c]
            out[grupo] = _logit(base) - _logit(self.pipeline.predict_proba(Xn)[:, 1])
        return out

    def prever(self, brutos: pd.DataFrame) -> list[dict]:
        """`brutos`: DataFrame com as colunas cruas (RAW_COLUMNS). Retorna uma lista de resultados."""
        df = brutos.copy()
        for c in RAW_COLUMNS:  # campos ausentes viram NaN e serão imputados (e reportados)
            if c not in df.columns:
                df[c] = np.nan
        prep, _ = preparar(df, deduplicar=False)  # inferência: uma resposta por pet recebido, na mesma ordem
        prep = prep.reset_index(drop=True)
        base = self._scores(prep)
        contribs = self._contribuicoes(prep, base)

        resultados = []
        for i, score in enumerate(base):
            linha = prep.iloc[i]
            positivas = {g: float(v[i]) for g, v in contribs.items() if v[i] > 0}
            total = sum(positivas.values())
            fatores = sorted(
                ({"grupo": g, "peso": round(c / total, 3), "motivo": _motivo(g, linha)}
                 for g, c in positivas.items() if total > 0 and c / total >= config.PESO_MINIMO),
                key=lambda f: f["peso"], reverse=True)[:config.MAX_FATORES]
            nivel = nivel_de_risco(float(score))
            if nivel == "NORMAL":
                acao = config.ACAO_NORMAL
            elif fatores:
                acao = config.ACOES[fatores[0]["grupo"]]
            else:
                acao = config.ACAO_FALLBACK
            imputados = [c for c in MODEL_COLUMNS if pd.isna(linha[c])]
            resultados.append({
                "score": round(float(score), 3),
                "nivel": nivel,
                "acao_recomendada": acao,
                "fatores": fatores,
                "campos_imputados": imputados,
                "modelo": self.meta["modelo"],
                "versao_modelo": self.meta["versao"],
            })
        return resultados


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    return Predictor()
