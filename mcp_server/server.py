"""Servidor MCP (somente leitura) que expõe o serviço de IA do VetFlow a qualquer host MCP
(Claude Desktop, n8n MCP Client...). Não existe nenhuma ferramenta de escrita: um agente
não consegue alterar dados clínicos por aqui.

Execução:  python mcp_server/server.py      (stdio)
Variáveis: VETFLOW_AI_URL (padrão http://localhost:8000), AI_SERVICE_API_KEY
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

BASE = os.getenv("VETFLOW_AI_URL", "http://localhost:8000").rstrip("/")
KEY = os.getenv("AI_SERVICE_API_KEY", "")

mcp = FastMCP("vetflow-risco")


def _get(path: str, **params) -> object:
    qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    req = urllib.request.Request(f"{BASE}{path}" + (f"?{qs}" if qs else ""))
    if KEY:
        req.add_header("X-API-Key", KEY)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


@mcp.tool()
def obter_risco_pet(pet_id: int) -> dict:
    """Score (0-1), nível (NORMAL/ATENÇÃO/CRÍTICO), fatores e ação recomendada de um pet. Somente leitura."""
    return _get(f"/demo/pets/{pet_id}")


@mcp.tool()
def listar_prioridade(nivel: str | None = None, limite: int = 10) -> list:
    """Pets ordenados do maior para o menor risco. `nivel` opcional: NORMAL, ATENÇÃO ou CRÍTICO."""
    return _get("/demo/pets", nivel=nivel, limit=max(1, min(limite, 50)))


@mcp.tool()
def info_modelo() -> dict:
    """Versão, algoritmo, métricas e avisos do modelo (inclui que os dados de treino são sintéticos)."""
    return _get("/model-info")


if __name__ == "__main__":
    mcp.run()
