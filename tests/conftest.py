"""Fixtures compartilhadas. Garante que dataset e modelo existam antes de testar."""
import pytest

from vetflow_ai import config
from vetflow_ai.data import main as gerar_dados
from vetflow_ai.train import main as treinar


@pytest.fixture(scope="session", autouse=True)
def artefatos():
    if not config.DATA_PATH.exists():
        gerar_dados()
    if not config.MODEL_PATH.exists():
        treinar()


@pytest.fixture()
def pet_em_dia():
    """Pet saudável e engajado: nada deveria disparar alerta."""
    return dict(
        especie="cachorro", porte="medio", idade_meses=48,
        dias_desde_ultima_vacina=120, intervalo_vacina_esperado_dias=365, vacinas_em_atraso=0,
        dias_desde_ultima_consulta=60, consultas_12m=4, faltas_consultas=0,
        tratamento_ativo=0, adesao_tratamento=None, doses_perdidas=0,
        alertas_comportamento=0, taxa_lembretes_ignorados=0.0, dias_desde_ultimo_login_app=2,
    )


@pytest.fixture()
def pet_grave():
    """Vacina muito atrasada, faltas, tratamento abandonado e tutor distante do app."""
    return dict(
        especie="cachorro", porte="grande", idade_meses=120,
        dias_desde_ultima_vacina=700, intervalo_vacina_esperado_dias=330, vacinas_em_atraso=3,
        dias_desde_ultima_consulta=400, consultas_12m=0, faltas_consultas=3,
        tratamento_ativo=1, adesao_tratamento=0.2, doses_perdidas=6,
        alertas_comportamento=2, taxa_lembretes_ignorados=0.9, dias_desde_ultimo_login_app=80,
    )
