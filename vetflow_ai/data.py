"""Gerador de dados SINTÉTICOS coerentes com a modelagem do VetFlow.

Por que sintético? Não há histórico real suficiente (nem autorização para usar
dados de tutores) para treinar o modelo. O gerador reproduz o formato das tabelas
do Oracle (pet, vacina, consulta, medicação, comportamento, uso do app) já
consolidadas em UMA linha por pet — exatamente o que a API Java entregaria ao
serviço de IA depois de consultar o banco.

Em produção o rótulo `risco_abandono_30d` viria do histórico real: o pet NÃO
cumpriu a vacina/retorno/dose prevista em até 30 dias após a data devida.
Aqui ele é simulado por um processo latente com ruído (nem tudo é determinístico).

O dataset sai propositalmente "sujo" (nulos, valores impossíveis) para que a
análise exploratória tenha o que limpar, como na aula do dataset Iris.

Uso:  python -m vetflow_ai.data
"""
import numpy as np
import pandas as pd

from .config import DATA_PATH, RANDOM_STATE, TARGET


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def generate_dataset(n: int = 1500, seed: int = RANDOM_STATE, dirty: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    especie = rng.choice(["cachorro", "gato"], n, p=[0.68, 0.32])
    porte = rng.choice(["pequeno", "medio", "grande"], n, p=[0.40, 0.38, 0.22])
    idade_meses = np.clip(rng.gamma(2.2, 30, n), 2, 216).round().astype(int)

    # Intervalo esperado entre vacinas: filhotes têm esquema mais curto, porte grande um pouco menor
    intervalo = np.where(idade_meses < 12, 120, np.where(porte == "grande", 330, 365))
    intervalo = intervalo + rng.integers(-20, 21, n)

    # Variável latente: "comprometimento" do tutor com o cuidado (0 = baixo, 1 = alto).
    # Ela cria correlações realistas entre atraso, faltas, adesão e engajamento.
    compromisso = rng.beta(4, 2, n)

    media_razao = 0.55 + 1.1 * (1 - compromisso)
    razao = np.clip(rng.gamma(5, media_razao / 5), 0.05, 3.5)
    dias_vacina = (razao * intervalo).round().astype(int)
    vacinas_atraso = np.clip(rng.poisson(np.clip(razao - 0.9, 0, None) * 2.2), 0, 8)

    dias_consulta = np.clip(rng.gamma(2, (60 + 260 * (1 - compromisso)) / 2), 3, 900).round().astype(int)
    consultas_12m = rng.poisson(1 + 2.5 * compromisso + (idade_meses > 96) * 1.0)
    faltas = rng.poisson(0.2 + 1.6 * (1 - compromisso))

    p_trat = 0.12 + 0.30 * (idade_meses >= 96)
    tratamento = (rng.random(n) < p_trat).astype(int)
    adesao = np.clip(rng.normal(0.45 + 0.5 * compromisso, 0.15), 0, 1)
    adesao = np.where(tratamento == 1, adesao.round(2), np.nan)
    doses_perdidas = np.where(tratamento == 1, rng.poisson(np.nan_to_num(1 - adesao) * 6), 0)

    alertas = rng.poisson(0.3, n)
    lembretes_ignorados = rng.binomial(8, np.clip(0.1 + 0.7 * (1 - compromisso), 0, 1)) / 8
    dias_login = np.clip(rng.gamma(1.5, (6 + 40 * (1 - compromisso)) / 1.5), 0, 365).round().astype(int)

    # Processo latente do rótulo (probabilidade de abandono do cuidado em 30 dias)
    adesao_full = np.nan_to_num(adesao, nan=1.0)
    z = (
        -3.6
        + 2.6 * np.clip(razao - 0.9, 0, 1.5)
        + 0.55 * vacinas_atraso
        + 0.9 * np.clip(dias_consulta - 180, 0, None) / 180
        + 0.5 * faltas
        + 2.2 * tratamento * (1 - adesao_full)
        + 1.2 * lembretes_ignorados
        + 0.8 * np.minimum(dias_login, 90) / 90
        + 0.35 * alertas
        + 1.0 * ((idade_meses >= 96) & (tratamento == 1) & (adesao_full < 0.6))  # idoso + tratamento + baixa adesão
        + 0.6 * ((idade_meses < 12) & (razao > 1.0))  # filhote com vacina atrasada
        + rng.normal(0, 0.7, n)
    )
    risco = (rng.random(n) < _sigmoid(z)).astype(int)

    df = pd.DataFrame(
        {
            "pet_id": np.arange(1, n + 1),
            "especie": especie,
            "porte": porte,
            "idade_meses": idade_meses,
            "dias_desde_ultima_vacina": dias_vacina,
            "intervalo_vacina_esperado_dias": intervalo,
            "vacinas_em_atraso": vacinas_atraso,
            "dias_desde_ultima_consulta": dias_consulta,
            "consultas_12m": consultas_12m,
            "faltas_consultas": faltas,
            "tratamento_ativo": tratamento,
            "adesao_tratamento": adesao,
            "doses_perdidas": doses_perdidas,
            "alertas_comportamento": alertas,
            "taxa_lembretes_ignorados": lembretes_ignorados,
            "dias_desde_ultimo_login_app": dias_login,
            TARGET: risco,
        }
    )

    if dirty:
        df = _inject_dirt(df, rng)
    return df


def _inject_dirt(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Simula erros reais de cadastro: nulos e valores impossíveis."""
    df = df.copy()
    n = len(df)

    # nulos aleatórios (falha de integração)
    df.loc[rng.random(n) < 0.03, "dias_desde_ultima_consulta"] = np.nan
    df.loc[rng.random(n) < 0.02, "taxa_lembretes_ignorados"] = np.nan
    df.loc[rng.random(n) < 0.015, "dias_desde_ultimo_login_app"] = np.nan

    # erro de digitação: idade em meses com um zero a mais (outlier)
    idx = rng.choice(n, size=max(1, int(0.006 * n)), replace=False)
    df.loc[idx, "idade_meses"] = df.loc[idx, "idade_meses"] * 10

    # datas invertidas: dias negativos
    idx = rng.choice(n, size=max(1, int(0.004 * n)), replace=False)
    df.loc[idx, "dias_desde_ultima_vacina"] = -df.loc[idx, "dias_desde_ultima_vacina"]
    return df


def main():
    df = generate_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset salvo em {DATA_PATH}  ({len(df)} pets)")
    print(f"Prevalência do rótulo {TARGET}: {df[TARGET].mean():.1%}")


if __name__ == "__main__":
    main()
