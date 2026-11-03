"""Testes da preparação e validação dos dados (limpeza, faixas, derivação de features)."""
import numpy as np
import pandas as pd

from vetflow_ai import config
from vetflow_ai.data import generate_dataset
from vetflow_ai.prep import MODEL_COLUMNS, checar_qualidade, derivar_features, marcar_fora_da_faixa, preparar


def test_gerador_e_reprodutivel():
    a, b = generate_dataset(n=200, seed=7), generate_dataset(n=200, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_dataset_sujo_realmente_tem_sujeira():
    df = generate_dataset(n=1500)
    assert (df["idade_meses"] > 300).any(), "esperava idades impossíveis"
    assert (df["dias_desde_ultima_vacina"] < 0).any(), "esperava dias negativos"
    assert df["dias_desde_ultima_consulta"].isna().any(), "esperava nulos"


def test_fora_da_faixa_vira_nan_e_e_contado():
    df = pd.DataFrame({"idade_meses": [24, 1820], "dias_desde_ultima_vacina": [-5, 100]})
    limpo, contagem = marcar_fora_da_faixa(df)
    assert np.isnan(limpo.loc[1, "idade_meses"]) and limpo.loc[0, "idade_meses"] == 24
    assert np.isnan(limpo.loc[0, "dias_desde_ultima_vacina"])
    assert contagem == {"idade_meses": 1, "dias_desde_ultima_vacina": 1}


def test_adesao_sem_tratamento_nao_e_falha():
    df = pd.DataFrame({"dias_desde_ultima_vacina": [100], "intervalo_vacina_esperado_dias": [365],
                       "tratamento_ativo": [0], "adesao_tratamento": [np.nan], "doses_perdidas": [np.nan]})
    r = derivar_features(df).iloc[0]
    assert r["adesao_tratamento"] == 1.0 and r["doses_perdidas"] == 0


def test_razao_de_vacina_e_relativa_ao_intervalo_do_pet():
    """Mesmos 200 dias: para um filhote (intervalo 120) é muito atrasado; para adulto (365) está em dia."""
    df = pd.DataFrame({"dias_desde_ultima_vacina": [200, 200], "intervalo_vacina_esperado_dias": [120, 365],
                       "tratamento_ativo": [0, 0], "adesao_tratamento": [np.nan] * 2, "doses_perdidas": [0, 0]})
    filhote, adulto = derivar_features(df)["razao_atraso_vacina"]
    assert filhote > 1.0 > adulto


def test_dataset_preparado_passa_na_validacao():
    prep, rel = preparar(generate_dataset(n=1500))
    assert checar_qualidade(prep) == []
    assert set(MODEL_COLUMNS) <= set(prep.columns)
    assert rel["valores_fora_da_faixa"], "a limpeza deveria ter marcado valores impossíveis"


def test_validacao_detecta_dado_ruim():
    prep, _ = preparar(generate_dataset(n=300))
    prep.loc[prep.index[0], "idade_meses"] = 9999
    assert any("idade_meses" in p for p in checar_qualidade(prep))
