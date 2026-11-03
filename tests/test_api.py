"""Testes da API: contrato, validação de entrada, lote, rotas de demonstração e autenticação."""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["modelo_carregado"] is True


def test_predict_devolve_contrato_completo(pet_grave):
    r = client.post("/predict", json={**pet_grave, "pet_id": 7, "nome_pet": "Thor"})
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["pet_id"] == 7 and corpo["nome_pet"] == "Thor"
    assert corpo["nivel"] == "CRÍTICO" and 0 <= corpo["score"] <= 1
    assert corpo["acao_recomendada"] and corpo["fatores"] and corpo["aviso"]
    assert {"grupo", "peso", "motivo"} <= set(corpo["fatores"][0])


@pytest.mark.parametrize("campo,valor", [
    ("idade_meses", 1820),            # erro de digitação
    ("idade_meses", -3),
    ("adesao_tratamento", 1.7),       # adesão > 100%
    ("dias_desde_ultima_vacina", -10),
    ("porte", "gigante"),
    ("especie", "papagaio"),
    ("tratamento_ativo", 2),
])
def test_dado_invalido_e_rejeitado_com_422(pet_em_dia, campo, valor):
    r = client.post("/predict", json={**pet_em_dia, campo: valor})
    assert r.status_code == 422


def test_campo_obrigatorio_ausente(pet_em_dia):
    del pet_em_dia["especie"]
    assert client.post("/predict", json=pet_em_dia).status_code == 422


def test_lote_resumo_soma_o_total(pet_em_dia, pet_grave):
    r = client.post("/predict/batch", json={"pets": [pet_em_dia, pet_grave, pet_grave]})
    assert r.status_code == 200
    corpo = r.json()
    assert corpo["total"] == 3 and sum(corpo["resumo"].values()) == 3
    assert corpo["resumo"]["CRÍTICO"] == 2 and corpo["resumo"]["NORMAL"] == 1


def test_lote_vazio_e_rejeitado():
    assert client.post("/predict/batch", json={"pets": []}).status_code == 422


def test_demo_lista_ordenada_por_risco_e_filtra_nivel():
    todos = client.get("/demo/pets?limit=20").json()
    scores = [p["score"] for p in todos]
    assert scores == sorted(scores, reverse=True) and len(todos) == 20
    criticos = client.get("/demo/pets?nivel=CRÍTICO&limit=50").json()
    assert criticos and all(p["nivel"] == "CRÍTICO" for p in criticos)
    assert all(p["nome_pet"] and p["tutor_nome"] for p in criticos)


@pytest.mark.parametrize("escrita", ["CRÍTICO", "CRITICO", "critico", " Crítico "])
def test_filtro_de_nivel_aceita_com_ou_sem_acento(escrita):
    """Agentes de IA costumam escrever o nível sem acento; a API deve entender."""
    r = client.get("/demo/pets", params={"nivel": escrita, "limit": 5})
    assert r.status_code == 200 and r.json() and all(p["nivel"] == "CRÍTICO" for p in r.json())


def test_demo_nivel_invalido_e_pet_inexistente():
    assert client.get("/demo/pets?nivel=BANANA").status_code == 422
    assert client.get("/demo/pets/999999").status_code == 404


def test_demo_pet_existente():
    pid = client.get("/demo/pets?limit=1").json()[0]["pet_id"]
    r = client.get(f"/demo/pets/{pid}")
    assert r.status_code == 200 and r.json()["pet_id"] == pid


def test_autenticacao_por_chave(monkeypatch, pet_em_dia):
    monkeypatch.setenv("AI_SERVICE_API_KEY", "segredo-de-teste")
    assert client.post("/predict", json=pet_em_dia).status_code == 401
    assert client.post("/predict", json=pet_em_dia, headers={"X-API-Key": "errada"}).status_code == 401
    assert client.post("/predict", json=pet_em_dia, headers={"X-API-Key": "segredo-de-teste"}).status_code == 200
    assert client.get("/health").status_code == 200  # health continua aberto para o orquestrador


def test_pet_sem_vacina_registrada_e_aceito_e_imputado():
    """Pet sem nenhuma vacina: o Oracle não tem o que informar; o modelo imputa e avisa."""
    r = client.post("/predict", json={"especie": "gato", "porte": "pequeno", "idade_meses": 30})
    assert r.status_code == 200
    assert "razao_atraso_vacina" in r.json()["campos_imputados"]
