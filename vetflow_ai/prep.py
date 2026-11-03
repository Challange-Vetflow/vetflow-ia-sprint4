"""Preparação dos dados: limpeza, validação e derivação de features.

Segue o roteiro da análise exploratória vista em aula (catalogar -> limpar ->
visualizar -> concluir). Esta etapa é o "limpar": a MESMA função é usada no treino
e na inferência, para o modelo nunca ver um dado tratado de forma diferente em produção.

Regras de limpeza (justificadas na documentação):
  * valores fora das faixas plausíveis (config.VALID_RANGES) viram NaN, não são "consertados";
  * poucos nulos (< 5%) em variável numérica com outliers -> imputação pela MEDIANA (no pipeline);
  * `adesao_tratamento` vazia sem tratamento ativo NÃO é falha: não há o que abandonar -> 1.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

RAW_COLUMNS = [
    "especie", "porte", "idade_meses",
    "dias_desde_ultima_vacina", "intervalo_vacina_esperado_dias", "vacinas_em_atraso",
    "dias_desde_ultima_consulta", "consultas_12m", "faltas_consultas",
    "tratamento_ativo", "adesao_tratamento", "doses_perdidas",
    "alertas_comportamento", "taxa_lembretes_ignorados", "dias_desde_ultimo_login_app",
]
MODEL_COLUMNS = config.NUMERIC + config.CATEGORICAL


def marcar_fora_da_faixa(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Troca valores impossíveis por NaN e conta quantos foram marcados por coluna."""
    df = df.copy()
    contagem: dict[str, int] = {}
    for col, (lo, hi) in config.VALID_RANGES.items():
        if col not in df.columns:
            continue
        invalido = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        if invalido.any():
            contagem[col] = int(invalido.sum())
            df.loc[invalido, col] = np.nan
    return df, contagem


def derivar_features(df: pd.DataFrame) -> pd.DataFrame:
    """A partir dos campos 'crus' (o que a API Java consolida do Oracle) calcula as features do modelo."""
    df = df.copy()
    # Personalização: atraso RELATIVO ao intervalo esperado daquele pet (filhote x adulto x porte)
    razao = df["dias_desde_ultima_vacina"] / df["intervalo_vacina_esperado_dias"]
    df["razao_atraso_vacina"] = razao.clip(0, 5)

    # Sem tratamento ativo: adesão ausente é esperado (nada a abandonar)
    sem_tratamento = df["tratamento_ativo"] == 0
    df.loc[sem_tratamento & df["adesao_tratamento"].isna(), "adesao_tratamento"] = 1.0
    df.loc[sem_tratamento & df["doses_perdidas"].isna(), "doses_perdidas"] = 0
    return df


def preparar(df: pd.DataFrame, deduplicar: bool = True) -> tuple[pd.DataFrame, dict]:
    """Pipeline completo de preparação. Retorna (df com colunas do modelo + extras, relatório).

    `deduplicar=True` só no TREINO. Na inferência nunca descartamos linhas: cada pet recebido
    precisa de exatamente uma resposta, na mesma ordem (e `pet_id` pode nem vir preenchido).
    """
    rel: dict = {"linhas_entrada": int(len(df))}
    if deduplicar and "pet_id" in df.columns:
        antes = len(df)
        df = df.drop_duplicates(subset="pet_id")
        rel["duplicados_removidos"] = antes - len(df)

    df, fora_faixa = marcar_fora_da_faixa(df)
    rel["valores_fora_da_faixa"] = fora_faixa
    df = derivar_features(df)

    nulos = df[MODEL_COLUMNS].isna().sum()
    rel["nulos_para_imputar"] = {k: int(v) for k, v in nulos.items() if v > 0}
    rel["linhas_saida"] = int(len(df))
    return df, rel


def checar_qualidade(df: pd.DataFrame) -> list[str]:
    """Testes de validação dos dados já preparados. Retorna lista de problemas (vazia = ok)."""
    problemas = []
    faltando = [c for c in MODEL_COLUMNS if c not in df.columns]
    if faltando:
        problemas.append(f"colunas ausentes: {faltando}")
        return problemas
    for col, (lo, hi) in config.VALID_RANGES.items():
        if col in df.columns:
            fora = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            if fora.any():
                problemas.append(f"{col}: {int(fora.sum())} valores fora de [{lo}, {hi}]")
    if not df["razao_atraso_vacina"].dropna().between(0, 5).all():
        problemas.append("razao_atraso_vacina fora de [0, 5]")
    if (df["tratamento_ativo"] == 0).any():
        sem = df[df["tratamento_ativo"] == 0]
        if sem["adesao_tratamento"].isna().any():
            problemas.append("adesao_tratamento ausente em pets sem tratamento")
    if df["especie"].isna().any() or df["porte"].isna().any():
        problemas.append("especie/porte com nulos")
    return problemas
