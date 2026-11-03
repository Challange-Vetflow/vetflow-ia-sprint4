"""Configurações centrais do serviço de IA do VetFlow.

Tudo que é "regra de negócio" do componente de IA (features, limiares de risco,
grupos de explicação e ações recomendadas) fica aqui, em um único lugar.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "pets_features.csv"
MODEL_PATH = ROOT / "models" / "vetflow_risk_model.joblib"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = ROOT / "docs" / "figures"

RANDOM_STATE = 42
TARGET = "risco_abandono_30d"

# --- Features usadas pelo modelo -------------------------------------------------
NUMERIC = [
    "idade_meses",
    "razao_atraso_vacina",
    "vacinas_em_atraso",
    "dias_desde_ultima_consulta",
    "consultas_12m",
    "faltas_consultas",
    "tratamento_ativo",
    "adesao_tratamento",
    "doses_perdidas",
    "alertas_comportamento",
    "taxa_lembretes_ignorados",
    "dias_desde_ultimo_login_app",
]
CATEGORICAL = ["especie", "porte"]

# Faixas plausíveis: valores fora delas são tratados como dado corrompido
# (viram NaN na limpeza do treino e geram erro 422 na API).
VALID_RANGES = {
    "idade_meses": (0, 300),
    "dias_desde_ultima_vacina": (0, 3650),
    "intervalo_vacina_esperado_dias": (30, 730),
    "vacinas_em_atraso": (0, 20),
    "dias_desde_ultima_consulta": (0, 3650),
    "consultas_12m": (0, 50),
    "faltas_consultas": (0, 50),
    "adesao_tratamento": (0.0, 1.0),
    "doses_perdidas": (0, 200),
    "alertas_comportamento": (0, 50),
    "taxa_lembretes_ignorados": (0.0, 1.0),
    "dias_desde_ultimo_login_app": (0, 3650),
}

# --- Níveis de risco (mesmo padrão NORMAL / ATENÇÃO / CRÍTICO das aulas) ----------
THRESHOLD_ATENCAO = 0.40
THRESHOLD_CRITICO = 0.70

# --- Explicação individual: grupos de features e ação recomendada ------------------
GRUPOS = {
    "Vacinação": ["razao_atraso_vacina", "vacinas_em_atraso"],
    "Tratamento": ["adesao_tratamento", "doses_perdidas"],
    "Acompanhamento clínico": [
        "dias_desde_ultima_consulta",
        "consultas_12m",
        "faltas_consultas",
    ],
    "Engajamento no app": ["taxa_lembretes_ignorados", "dias_desde_ultimo_login_app"],
    "Comportamento": ["alertas_comportamento"],
}

ACOES = {
    "Vacinação": "Agendar reforço vacinal",
    "Tratamento": "Contato de acompanhamento do tratamento",
    "Acompanhamento clínico": "Agendar consulta de retorno",
    "Engajamento no app": "Reengajar o tutor com lembrete personalizado",
    "Comportamento": "Avaliação comportamental na próxima consulta",
}
ACAO_NORMAL = "Nenhuma ação necessária — manter lembretes de rotina"
ACAO_FALLBACK = "Acompanhamento de rotina"

# Peso mínimo (parcela do risco do pet explicada por um grupo) para citá-lo como fator principal.
# O peso é calculado em log-odds: em escala de probabilidade o score satura perto de 1,0 e um grupo
# isolado pareceria "não pesar" justamente nos casos mais graves.
PESO_MINIMO = 0.15
MAX_FATORES = 3
