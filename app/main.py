"""API do serviço de IA do VetFlow (FastAPI).

Quem chama:
  * a API Java (Spring Boot): monta os dados do pet no Oracle e chama POST /predict;
  * o n8n (agente de IA): usa os endpoints como ferramentas (tools);
  * o servidor MCP (mcp_server/): expõe as mesmas consultas, somente leitura, para agentes.

Segurança: se a variável de ambiente AI_SERVICE_API_KEY estiver definida, todas as rotas
(exceto /health) exigem o cabeçalho X-API-Key. Nenhuma chave fica no código.
"""
from __future__ import annotations

import os
import secrets
from functools import lru_cache
from typing import Optional

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query

from vetflow_ai import config
from vetflow_ai.predictor import get_predictor

from .schemas import LoteEntrada, LoteSaida, PetDemo, PetFeatures, Previsao

app = FastAPI(
    title="VetFlow — Serviço de IA",
    version="1.0.0",
    description="Score de risco de abandono do cuidado do pet, nível de alerta (NORMAL/ATENÇÃO/CRÍTICO), "
                "motivos e ação recomendada.",
)

NOMES_PET = ["Thor", "Mel", "Bidu", "Luna", "Rex", "Nina", "Bob", "Mia", "Zeca", "Lola", "Duque", "Chiara", "Pipoca", "Amora", "Toby"]
NOMES_TUTOR = ["Ana", "Carlos", "Beatriz", "Rafael", "Juliana", "Marcos", "Patrícia", "Lucas", "Fernanda", "Diego"]


def verificar_chave(x_api_key: Optional[str] = Header(default=None)) -> None:
    esperada = os.getenv("AI_SERVICE_API_KEY")
    if not esperada:  # modo desenvolvimento: sem chave configurada, sem autenticação
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, esperada):
        raise HTTPException(status_code=401, detail="Chave de API inválida ou ausente (X-API-Key).")


Auth = Depends(verificar_chave)


def _resultado(entrada: PetFeatures, previsao: dict) -> Previsao:
    return Previsao(pet_id=entrada.pet_id, nome_pet=entrada.nome_pet, **previsao)


@app.get("/health")
def health():
    try:
        get_predictor()
        return {"status": "ok", "modelo_carregado": True}
    except Exception as exc:  # modelo ausente: serviço de pé, mas não pronto
        raise HTTPException(status_code=503, detail=f"Modelo não carregado: {exc}")


@app.get("/model-info", dependencies=[Auth])
def model_info():
    meta = get_predictor().meta
    return {k: meta[k] for k in ("versao", "modelo", "treinado_em", "n_treino_total", "limiares", "dados_sinteticos")}


@app.post("/predict", response_model=Previsao, dependencies=[Auth])
def predict(pet: PetFeatures):
    previsao = get_predictor().prever(pd.DataFrame([pet.model_dump()]))[0]
    return _resultado(pet, previsao)


@app.post("/predict/batch", response_model=LoteSaida, dependencies=[Auth])
def predict_batch(lote: LoteEntrada):
    previsoes = get_predictor().prever(pd.DataFrame([p.model_dump() for p in lote.pets]))
    resultados = [_resultado(p, r) for p, r in zip(lote.pets, previsoes)]
    resumo = {n: sum(r.nivel == n for r in resultados) for n in ("NORMAL", "ATENÇÃO", "CRÍTICO")}
    return LoteSaida(total=len(resultados), resumo=resumo, resultados=resultados)


# ------------------------------------------------------------------------------------------
# Rotas de DEMONSTRAÇÃO: fazem o papel do Oracle/API Java enquanto a integração real não existe.
# Leem o dataset sintético, passam pela MESMA limpeza e devolvem o ranking. Em produção, o n8n e
# o app consultam a API Java, que chama POST /predict com os dados reais do banco.
# ------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _ranking_demo() -> list[dict]:
    bruto = pd.read_csv(config.DATA_PATH)
    previsoes = get_predictor().prever(bruto)
    itens = []
    for (_, linha), prev in zip(bruto.iterrows(), previsoes):
        pid = int(linha["pet_id"])
        itens.append({
            "pet_id": pid,
            "nome_pet": NOMES_PET[pid % len(NOMES_PET)],
            "tutor_nome": NOMES_TUTOR[(pid * 7) % len(NOMES_TUTOR)],
            **prev,
        })
    return sorted(itens, key=lambda i: i["score"], reverse=True)


_NIVEIS = {"NORMAL": "NORMAL", "ATENCAO": "ATENÇÃO", "ATENÇÃO": "ATENÇÃO", "CRITICO": "CRÍTICO", "CRÍTICO": "CRÍTICO"}


def _normalizar_nivel(valor: Optional[str]) -> Optional[str]:
    """Aceita com ou sem acento e em qualquer caixa (agentes de IA costumam escrever 'CRITICO')."""
    if valor is None:
        return None
    nivel = _NIVEIS.get(valor.strip().upper())
    if nivel is None:
        raise HTTPException(status_code=422, detail="nivel deve ser NORMAL, ATENÇÃO ou CRÍTICO (com ou sem acento)")
    return nivel


@app.get("/demo/pets", response_model=list[PetDemo], dependencies=[Auth], tags=["demo"])
def demo_lista(
    limit: int = Query(10, ge=1, le=100),
    nivel: Optional[str] = Query(None, description="NORMAL, ATENÇÃO ou CRÍTICO (aceita sem acento)"),
):
    """Lista de prioridade (maior risco primeiro). Use `nivel=CRITICO` para a fila de contato da clínica."""
    alvo = _normalizar_nivel(nivel)
    itens = _ranking_demo()
    if alvo:
        itens = [i for i in itens if i["nivel"] == alvo]
    return itens[:limit]


@app.get("/demo/pets/{pet_id}", response_model=PetDemo, dependencies=[Auth], tags=["demo"])
def demo_pet(pet_id: int):
    for item in _ranking_demo():
        if item["pet_id"] == pet_id:
            return item
    raise HTTPException(status_code=404, detail=f"Pet {pet_id} não encontrado")
