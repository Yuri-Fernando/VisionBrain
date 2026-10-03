# Contribuindo

Contribuições devem preservar os princípios do projeto: contratos estreitos, comportamento offline testável, ausência de segredos e separação entre loop de vídeo e integrações externas.

## Ambiente

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Antes de enviar uma mudança

```powershell
python -m ruff check src tests apps
python -m pytest -q
```

Novos backends de câmera ou inferência devem implementar os protocolos existentes, incluir testes sem hardware quando possível e documentar claramente o que foi ou não validado em dispositivo real.
