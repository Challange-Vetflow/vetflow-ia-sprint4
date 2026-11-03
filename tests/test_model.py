"""Testes do modelo e dos cenários de uso (validação em situações reais de clínica)."""
import pandas as pd
import pytest

from vetflow_ai import config
from vetflow_ai.predictor import get_predictor, nivel_de_risco
from vetflow_ai.prep import MODEL_COLUMNS, preparar
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def prever(dados: dict) -> dict:
    return get_predictor().prever(pd.DataFrame([dados]))[0]


def test_limiares_de_nivel():
    assert nivel_de_risco(0.39) == "NORMAL"
    assert nivel_de_risco(0.40) == "ATENÇÃO"
    assert nivel_de_risco(0.69) == "ATENÇÃO"
    assert nivel_de_risco(0.70) == "CRÍTICO"


def test_modelo_tem_poder_preditivo_minimo():
    """O modelo publicado precisa ranquear bem pets de risco (AUC em dado separado)."""
    prep, _ = preparar(pd.read_csv(config.DATA_PATH))
    _, X_te, _, y_te = train_test_split(prep[MODEL_COLUMNS], prep[config.TARGET], test_size=0.25,
                                        stratify=prep[config.TARGET], random_state=config.RANDOM_STATE)
    scores = get_predictor().pipeline.predict_proba(X_te)[:, 1]
    assert roc_auc_score(y_te, scores) >= 0.75


def test_pet_em_dia_fica_normal(pet_em_dia):
    r = prever(pet_em_dia)
    assert r["nivel"] == "NORMAL" and r["score"] < config.THRESHOLD_ATENCAO
    assert r["acao_recomendada"] == config.ACAO_NORMAL


def test_caso_grave_fica_critico(pet_grave):
    r = prever(pet_grave)
    assert r["nivel"] == "CRÍTICO" and r["score"] >= config.THRESHOLD_CRITICO
    assert r["fatores"], "um caso crítico precisa de ao menos um motivo"


def test_mais_faltas_nunca_reduz_o_risco(pet_em_dia):
    scores = [prever({**pet_em_dia, "faltas_consultas": f})["score"] for f in range(0, 5)]
    assert scores == sorted(scores)


def test_pior_adesao_aumenta_o_risco_em_tratamento(pet_em_dia):
    base = {**pet_em_dia, "tratamento_ativo": 1}
    boa, ruim = prever({**base, "adesao_tratamento": 0.95}), prever({**base, "adesao_tratamento": 0.2, "doses_perdidas": 6})
    assert ruim["score"] > boa["score"]


def test_personalizacao_acao_muda_conforme_o_motivo(pet_em_dia):
    """Mesmo nível de alerta, causas diferentes -> ações diferentes."""
    so_vacina = {**pet_em_dia, "dias_desde_ultima_vacina": 900, "vacinas_em_atraso": 4, "faltas_consultas": 1}
    so_tratamento = {**pet_em_dia, "tratamento_ativo": 1, "adesao_tratamento": 0.1, "doses_perdidas": 8,
                     "taxa_lembretes_ignorados": 0.8}
    a, b = prever(so_vacina), prever(so_tratamento)
    assert a["fatores"][0]["grupo"] == "Vacinação" and a["acao_recomendada"] == config.ACOES["Vacinação"]
    assert b["fatores"][0]["grupo"] == "Tratamento" and b["acao_recomendada"] == config.ACOES["Tratamento"]


def test_filhote_e_adulto_com_mesmo_atraso_em_dias_tem_scores_diferentes(pet_em_dia):
    """200 dias sem vacina: para filhote (esquema de 120 dias) é atraso; para adulto não."""
    adulto = {**pet_em_dia, "dias_desde_ultima_vacina": 200, "intervalo_vacina_esperado_dias": 365, "idade_meses": 48}
    filhote = {**adulto, "intervalo_vacina_esperado_dias": 120, "idade_meses": 5}
    assert prever(filhote)["score"] > prever(adulto)["score"]


def test_dado_ausente_e_imputado_e_reportado(pet_em_dia):
    r = prever({**pet_em_dia, "dias_desde_ultima_consulta": None, "taxa_lembretes_ignorados": None})
    assert {"dias_desde_ultima_consulta", "taxa_lembretes_ignorados"} <= set(r["campos_imputados"])
    assert 0 <= r["score"] <= 1


def test_dado_corrompido_nao_gera_decisao_absurda(pet_em_dia):
    """Idade de 1820 meses (erro de digitação) vira 'ausente' e é imputada; o score continua válido."""
    r = prever({**pet_em_dia, "idade_meses": 1820})
    assert "idade_meses" in r["campos_imputados"] and r["nivel"] == "NORMAL"


def test_scores_do_lote_estao_em_zero_um():
    res = get_predictor().prever(pd.read_csv(config.DATA_PATH).head(200))
    assert all(0 <= r["score"] <= 1 for r in res)
    assert {r["nivel"] for r in res} <= {"NORMAL", "ATENÇÃO", "CRÍTICO"}
