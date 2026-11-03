"""Contrato (JSON) do serviço de IA. As faixas vêm de config.VALID_RANGES: dado fora delas é rejeitado (422)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from vetflow_ai import config


def _faixa(campo: str) -> dict:
    lo, hi = config.VALID_RANGES[campo]
    return {"ge": lo, "le": hi}


class PetFeatures(BaseModel):
    """Dados consolidados de UM pet, como a API Java os monta a partir do Oracle."""

    pet_id: Optional[int] = Field(None, description="Identificador do pet (apenas devolvido na resposta)")
    nome_pet: Optional[str] = Field(None, max_length=80)

    especie: Literal["cachorro", "gato"]
    porte: Literal["pequeno", "medio", "grande"]
    idade_meses: int = Field(..., **_faixa("idade_meses"))

    # Pet sem nenhuma vacina registrada: os dois campos chegam ausentes, o modelo imputa a mediana
    # e a resposta os lista em `campos_imputados` (o motivo deixa de citar vacina).
    dias_desde_ultima_vacina: Optional[int] = Field(None, **_faixa("dias_desde_ultima_vacina"))
    intervalo_vacina_esperado_dias: Optional[int] = Field(None, **_faixa("intervalo_vacina_esperado_dias"))
    vacinas_em_atraso: int = Field(0, **_faixa("vacinas_em_atraso"))

    dias_desde_ultima_consulta: Optional[int] = Field(None, **_faixa("dias_desde_ultima_consulta"))
    consultas_12m: int = Field(0, **_faixa("consultas_12m"))
    faltas_consultas: int = Field(0, **_faixa("faltas_consultas"))

    tratamento_ativo: Literal[0, 1] = 0
    adesao_tratamento: Optional[float] = Field(None, **_faixa("adesao_tratamento"))
    doses_perdidas: int = Field(0, **_faixa("doses_perdidas"))

    alertas_comportamento: int = Field(0, **_faixa("alertas_comportamento"))
    taxa_lembretes_ignorados: Optional[float] = Field(None, **_faixa("taxa_lembretes_ignorados"))
    dias_desde_ultimo_login_app: Optional[int] = Field(None, **_faixa("dias_desde_ultimo_login_app"))

    model_config = {
        "json_schema_extra": {
            "example": {
                "pet_id": 101, "nome_pet": "Thor", "especie": "cachorro", "porte": "grande", "idade_meses": 84,
                "dias_desde_ultima_vacina": 430, "intervalo_vacina_esperado_dias": 330, "vacinas_em_atraso": 2,
                "dias_desde_ultima_consulta": 260, "consultas_12m": 1, "faltas_consultas": 2,
                "tratamento_ativo": 1, "adesao_tratamento": 0.35, "doses_perdidas": 5,
                "alertas_comportamento": 1, "taxa_lembretes_ignorados": 0.6, "dias_desde_ultimo_login_app": 40,
            }
        }
    }


class Fator(BaseModel):
    grupo: str
    peso: float = Field(..., ge=0, le=1, description="Parcela do risco deste pet explicada pelo grupo (soma dos fatores citados <= 1)")
    motivo: str = Field(..., description="Frase com os fatos do pet que explicam o fator")


class Previsao(BaseModel):
    pet_id: Optional[int] = None
    nome_pet: Optional[str] = None
    score: float = Field(..., ge=0, le=1, description="Probabilidade estimada de abandono do cuidado em 30 dias")
    nivel: Literal["NORMAL", "ATENÇÃO", "CRÍTICO"]
    acao_recomendada: str
    fatores: list[Fator]
    campos_imputados: list[str] = Field(default_factory=list, description="Campos ausentes/inválidos preenchidos pela mediana")
    modelo: str
    versao_modelo: str
    aviso: str = "Apoio à decisão: a avaliação clínica é sempre do médico-veterinário."


class LoteEntrada(BaseModel):
    pets: list[PetFeatures] = Field(..., min_length=1, max_length=500)


class LoteSaida(BaseModel):
    total: int
    resumo: dict[str, int]
    resultados: list[Previsao]


class PetDemo(Previsao):
    tutor_nome: Optional[str] = None
