# VetFlow — Serviço de IA: risco de abandono de cuidados

Parte de IA do projeto **CLYVO VET / VetFlow** (Sprint 4, FIAP). Prevê, para cada pet, o risco de o tutor **abandonar vacinas, retornos ou tratamento nos próximos 30 dias**, explica o motivo e sugere a próxima ação. O score vem de um **modelo estatístico determinístico**; o LLM (n8n) só redige mensagens.

> ⚠️ **Dados sintéticos.** As métricas validam a **arquitetura** ponta a ponta, não o desempenho em clínicas reais.

## Equipe
| Nome | RM | Turma |
|---|---|---|
| Andrei de Paiva Gibbini | 563061 | 2TDSPF |
| Pedro Sakai Silva Zambaca | 565956 | 2TDSPF |
| Pedro Santos Pequini | 561842 | 2TDSPF |
| Arthur Câmara | 562310 | 2TDSPG |
| Diogo Cunha | 563654 | 2TDSPF |

## Arquitetura
```
App React Native ──► API Java (Spring) ──► Serviço de IA (FastAPI) ──► modelo .joblib
                          │  ▲                    ▲
                    Oracle (view)          n8n (agente) / MCP (leitura)
                                               │
                                  Telegram (tutor) · e-mail (clínica)
```

## Estrutura
| Pasta | Conteúdo |
|---|---|
| `vetflow_ai/` | configuração, geração de dados, preparação, treino, previsão com explicações |
| `app/` | API FastAPI (`/predict`, `/predict/batch`, `/model-info`, `/health`, `/demo/*`) |
| `notebooks/` | `01_eda_vetflow.ipynb` — análise exploratória (roda local e no Colab) |
| `reports/`, `docs/figures/` | métricas e gráficos gerados pelo treino |
| `tests/` | 39 testes (preparação, modelo, API) |
| `n8n/` | fluxo do agente + roteamento NORMAL/ATENÇÃO/CRÍTICO |
| `mcp_server/` | servidor MCP somente leitura |
| `data/` | dataset sintético (gerado por `vetflow_ai.data`) |
| `models/` | modelo treinado (`.joblib`) |
| `Dockerfile` | imagem do serviço (modelo treinado no build) |
| `docs/` | integração com Java e app, guia de deploy na Oracle Cloud, figuras |

## Como usar
```bash
pip install -r requirements-dev.txt
python -m vetflow_ai.data          # gera data/pets_features.csv
python -m vetflow_ai.train         # compara 4 modelos, salva models/ e reports/
pytest                             # 39 testes
export AI_SERVICE_API_KEY=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
uvicorn app.main:app --port 8000   # docs em http://localhost:8000/docs
```
Exemplo:
```bash
curl -X POST localhost:8000/predict -H "X-API-Key: $AI_SERVICE_API_KEY" -H "Content-Type: application/json" \
  -d '{"especie":"cachorro","porte":"medio","idade_meses":48,"dias_desde_ultima_vacina":500,"intervalo_vacina_esperado_dias":365,"vacinas_em_atraso":2,"dias_desde_ultima_consulta":400,"consultas_12m":0,"faltas_consultas":2,"tratamento_ativo":0,"alertas_comportamento":1,"taxa_lembretes_ignorados":0.8,"dias_desde_ultimo_login_app":90}'
```
Docker: `docker build -t vetflow-ai . && docker run -p 8000:8000 -e AI_SERVICE_API_KEY=... vetflow-ai` (o modelo é treinado no build). Implantação na Oracle Cloud: [`docs/deploy-oracle-cloud.md`](docs/deploy-oracle-cloud.md).

A API Java (`vetflow-java`) e o aplicativo (`VetFlow-MobileApp`) consomem este serviço. Veja [`docs/integracao.md`](docs/integracao.md).

## Tecnologias
Python, pandas, scikit-learn, FastAPI/Pydantic, pytest, Docker, n8n (OpenAI), MCP. Integrações: Oracle, Spring Boot, React Native.

## Resultados parciais (dados sintéticos, 1500 pets, teste de 375)
| Modelo | AUC (CV 5-fold) | AUC teste | F1 |
|---|---|---|---|
| **Regressão logística** ✅ | 0.832 ± 0.016 | 0.850 | 0.617 |
| Random forest | 0.826 ± 0.025 | 0.838 | 0.584 |
| Gradient boosting | 0.808 ± 0.024 | 0.836 | 0.575 |
| MLP | 0.822 ± 0.010 | 0.832 | 0.609 |

Taxa real de risco por nível no teste: NORMAL 12,8% · ATENÇÃO 59,4% · CRÍTICO 84,6%. Os quatro modelos ficaram a menos de 0,03 de AUC entre si (diferença não conclusiva); o rótulo sintético é quase linear, o que favorece a regressão logística. Detalhes em [`reports/avaliacao_modelos.md`](reports/avaliacao_modelos.md).

## Personalização e acompanhamento contínuo
- O atraso de vacina é medido **relativo ao intervalo esperado do próprio pet** (filhote ≠ adulto ≠ porte).
- Cada previsão traz **fatores** com motivo em linguagem natural (só com valores reais) e **ação recomendada**.
- O n8n roda diariamente e aciona o tutor (ATENÇÃO/CRÍTICO) e a clínica (CRÍTICO).

## Segurança e limitações
- Entrada validada por faixas (422 em dado corrompido); valores impossíveis viram nulo e são imputados pela mediana; os campos imputados são informados na resposta.
- Chave `X-API-Key` opcional/obrigatória por variável de ambiente; segredos fora do Git (`.env.example`).
- O LLM **não** calcula risco nem decide nível; "NUNCA invente valores" no prompt e fallback determinístico se a resposta falhar.
- MCP somente leitura. A decisão clínica é sempre do veterinário.
