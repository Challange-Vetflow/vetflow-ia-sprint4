# Implantação do serviço de IA na Oracle Cloud (Docker)

Passo a passo para colocar o serviço FastAPI (`Dockerfile` da raiz) em uma VM da Oracle Cloud.

## 1. Criar a VM
1. Console Oracle Cloud → *Compute* → *Instances* → *Create instance*.
2. Imagem: **Ubuntu 22.04**. Shape: `VM.Standard.A1.Flex` (ARM, Always Free: 1 a 2 OCPU e 6 a 12 GB) é a opção recomendada; `VM.Standard.E2.1.Micro` (1 GB) é pouco para instalar o scikit-learn no build.
3. Rede: mantenha **IP público** e baixe a chave SSH.

## 2. Liberar a porta 8000
- No console: *Networking* → sua VCN → *Security Lists* → regra de entrada TCP, porta **8000**, origem `0.0.0.0/0` (a proteção é a chave `X-API-Key`; se souber o IP de quem consome, restrinja).
- Dentro da VM (as imagens da Oracle trazem regras de firewall próprias):
```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 8000 -j ACCEPT
sudo netfilter-persistent save
```

## 3. Instalar Docker e subir o serviço
```bash
ssh -i sua-chave.key ubuntu@IP_PUBLICO
sudo apt-get update && sudo apt-get install -y docker.io git
sudo usermod -aG docker $USER && newgrp docker

git clone https://github.com/Challange-Vetflow/vetflow-ia.git vetflow-ai
cd vetflow-ai
docker build -t vetflow-ai .

export AI_SERVICE_API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
echo "$AI_SERVICE_API_KEY"        # guarde: Java e n8n usam esta chave
docker run -d --name vetflow-ai --restart unless-stopped -p 8000:8000 \
  -e AI_SERVICE_API_KEY="$AI_SERVICE_API_KEY" vetflow-ai
```
O modelo é treinado durante o `docker build` (dados sintéticos), então a imagem sempre carrega um modelo compatível com as bibliotecas instaladas.

## 4. Testar
```bash
curl http://localhost:8000/health
curl -H "X-API-Key: $AI_SERVICE_API_KEY" "http://localhost:8000/demo/pets?limit=3"
```
De fora da VM, troque `localhost` pelo IP público. `/docs` mostra o Swagger.

## 5. Conectar os outros componentes
| Quem | O que configurar |
|---|---|
| API Java (`vetflow-java`) | variáveis `VETFLOW_AI_URL=http://IP_PUBLICO:8000` e `AI_SERVICE_API_KEY` |
| n8n cloud | nó *Config* → `ai_url`; credencial Header Auth `X-API-Key` |
| Servidor MCP (`mcp_server/`) | `VETFLOW_AI_URL` e `AI_SERVICE_API_KEY` |

## 6. HTTPS (opcional, recomendado)
Sem TLS, a chave trafega em texto claro. Com um domínio (ou `IP-COM-TRACOS.nip.io`) e as portas 80/443 liberadas, o Caddy resolve em poucos comandos:
```bash
sudo apt-get install -y caddy
echo 'SEU-DOMINIO { reverse_proxy localhost:8000 }' | sudo tee /etc/caddy/Caddyfile
sudo systemctl reload caddy
```
Depois use `https://SEU-DOMINIO` em `VETFLOW_AI_URL` e no n8n.

## Cuidados
- As rotas `/demo/*` devolvem dados fictícios para demonstração; em produção real, a API Java envia os dados de cada pet em `POST /predict`.
- O treino usa dados sintéticos: o serviço informa isso em `/model-info`.
