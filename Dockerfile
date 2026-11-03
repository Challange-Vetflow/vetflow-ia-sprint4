FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vetflow_ai ./vetflow_ai
COPY app ./app

# Gera o dataset sintético e treina o modelo NO BUILD: assim o arquivo .joblib sempre é
# compatível com as versões de scikit-learn instaladas na imagem.
RUN python -m vetflow_ai.data && python -m vetflow_ai.train

RUN useradd --create-home appuser && chown -R appuser /srv
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# AI_SERVICE_API_KEY deve ser definida no deploy (docker run -e AI_SERVICE_API_KEY=...)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
