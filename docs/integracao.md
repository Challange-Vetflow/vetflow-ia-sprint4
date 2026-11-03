# Integração do serviço de IA com o VetFlow

Este serviço é consumido pela API Java e, por meio dela, pelo aplicativo e pela web. O código de cada lado fica no seu próprio repositório:

| Parte | Repositório | O que foi adicionado |
|---|---|---|
| API Java (Spring Boot) | `vetflow-java` | `AiRiskClient`, `RiscoService`, `PetFeaturesRepository`, rota `GET /api/pets/{id}/risco`, cartão no prontuário (Thymeleaf), scripts Oracle 07 a 09 |
| Aplicativo (React Native / Expo) | `VetFlow-MobileApp` | `riscoService`, hook `useRisco`, componente `RiscoCard`, cartão na tela de detalhe do pet |

## Fluxo de uma consulta
```
Tutor abre o pet no app ou na web
        │
        ▼
API Java  GET /api/pets/{id}/risco        (sessão e permissões do VetFlow)
        │  lê a view Oracle VW_PET_FEATURES (uma linha por pet)
        ▼
Serviço de IA  POST /predict              (cabeçalho X-API-Key)
        │  prepara os dados, calcula o score e explica
        ▼
Resposta: score, nível, ação recomendada, fatores, campos estimados
        │
        ▼
App e web exibem o cartão "Acompanhamento inteligente"
```

## Dados que o Oracle entrega por pet (view `VW_PET_FEATURES`)
`especie`, `porte`, `idade_meses`, `dias_desde_ultima_vacina`, `intervalo_vacina_esperado_dias`, `vacinas_em_atraso`, `dias_desde_ultima_consulta`, `consultas_12m`, `faltas_consultas`, `tratamento_ativo`, `taxa_lembretes_ignorados`, `dias_desde_ultimo_login_app`.
Campos sem fonte no banco seguem como nulo: o serviço os estima pela mediana e informa em `campos_imputados`.

## Resposta da API Java (exemplo)
```json
{
  "score": 0.983,
  "nivel": "CRÍTICO",
  "acaoRecomendada": "Agendar reforço vacinal",
  "fatores": [
    { "grupo": "Vacinação", "peso": 0.38, "motivo": "última vacina com 1,6x o intervalo esperado, 2 vacina(s) em atraso" },
    { "grupo": "Engajamento no app", "peso": 0.368, "motivo": "83% dos lembretes ignorados, sem abrir o app há 120 dias" }
  ],
  "camposImputados": ["adesao_tratamento"]
}
```

## Comportamento em situações especiais
| Situação | Resposta |
|---|---|
| Serviço de IA fora do ar | A API Java devolve 503; app e web continuam funcionando, sem o cartão |
| Dado fora da faixa | O serviço rejeita com 422 |
| Espécie fora do modelo (só cães e gatos) | 400; o cartão não aparece |
| Tutor sem permissão sobre o pet | 403 ou 404, conforme as regras do VetFlow |

## Acompanhamento contínuo
O fluxo do n8n (`n8n/`) consulta este mesmo serviço todo dia, roteia por nível e usa um agente de IA para redigir a mensagem ao tutor. O servidor MCP (`mcp_server/`) expõe o serviço em modo somente leitura.

## Configuração
| Variável | Onde | Para quê |
|---|---|---|
| `AI_SERVICE_API_KEY` | serviço de IA, API Java, n8n, MCP | chave enviada em `X-API-Key` |
| `VETFLOW_AI_URL` | API Java, MCP | endereço do serviço de IA |
