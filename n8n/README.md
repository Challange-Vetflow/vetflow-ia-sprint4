# Fluxo n8n — VetFlow (risco de abandono → contato)

Arquivo: `vetflow_workflow.json` (importar em n8n cloud: *Workflows → Import from file*).

## O que o fluxo faz
1. **Gatilho** manual ou diário às 8h.
2. **Config**: URL do serviço de IA e quantos pets buscar.
3. **Buscar pets em risco**: `GET /demo/pets` (lista ordenada por risco; em produção, a API Java).
4. **Roteamento por nível** (decidido pelo modelo, não pelo LLM): CRÍTICO / ATENÇÃO / NORMAL.
5. **Agente de contato** (OpenAI + ferramenta `consultar_risco_pet`): só **redige** a mensagem ao tutor e o resumo para a clínica. System prompt com "NUNCA invente valores".
6. **Interpretar resposta**: se o LLM não devolver JSON válido, usa mensagem padrão montada só com dados do modelo.
7. **Telegram ao tutor** (ATENÇÃO e CRÍTICO) e **e-mail à clínica** (só CRÍTICO).

## Antes de rodar (configurar no n8n)
- Credencial **OpenAI** (sua API key).
- Credencial **Header Auth** `X-API-Key` = valor de `AI_SERVICE_API_KEY` do serviço.
- Credencial **Telegram** (bot do @BotFather) e trocar `SEU_CHAT_ID_DE_TESTE` pelo seu chat.
- Credencial **SMTP** e trocar os e-mails de exemplo.
- Em **Config**, trocar `ai_url` pela URL pública do serviço (n8n cloud não enxerga `localhost`; use deploy ou um túnel como ngrok).

## Honestidade
- O JSON foi gerado por script e validado como JSON/conexões, mas **não foi importado numa instância real**. Se algum nó acusar parâmetro diferente na sua versão do n8n, ajuste pela interface (a estrutura do fluxo se mantém).
- O LLM não calcula risco nem decide o nível: isso é do modelo determinístico. Decisão clínica é do veterinário.
- Extensão possível: supervisor + agentes especialistas (Vacinas / Tratamento / Retornos), cada um com sua ferramenta.
