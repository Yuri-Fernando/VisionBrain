# VisionBrain Dashboard e fluxo end-to-end

Esta camada mantém a interface fora do loop de câmera/inferência. O engine grava eventos e evidências; o dashboard local apenas lê esses artefatos.

## Instalação

Use Python 3.11 ou superior:

```powershell
py -3.11 -m pip install -e ".[dashboard,notebook,dev]"
```

O extra `yolo` não é necessário para o demo sintético.

## Demo E2E sem webcam

```powershell
visionbrain demo --output outputs/demo --frames 24
```

Artefatos esperados:

- `outputs/demo/events.jsonl` — audit trail;
- `outputs/demo/snapshots/` — evidências anotadas;
- `outputs/demo/summary.json` — resumo da execução.

Também existe um runtime sem detector/model download:

```powershell
visionbrain run --config config/synthetic_demo.yaml --frames 30 --no-display
```

## Dashboard

```powershell
visionbrain dashboard
```

Na barra lateral, informe `outputs/demo/events.jsonl` para visualizar o demo. O painel oferece KPIs, distribuição, linha do tempo, filtros, tabela e evidence viewer.

O dashboard é local e somente leitura. Ele não abre câmera, não chama webhook e não inicia inferência.

## Notebook

Abra `notebooks/visionbrain_end_to_end.ipynb` e execute de cima para baixo. O notebook usa uma fonte sintética determinística e não requer webcam, GPU, pesos YOLO, VLM, n8n ou conexão externa.

## Fonte industrial

O runtime agora escolhe a fonte por factory. Para uma câmera GenICam/GenTL, instale o extra e informe o CTI do fabricante:

```yaml
camera:
  source: "genicam:C:/caminho/fabricante.cti"
```

Esse caminho está integrado ao contrato do runtime, mas só pode ser considerado validado depois de testes com o dispositivo e o GenTL Producer reais.

## Limites honestos

- o demo simula deteções rastreadas; ele valida a orquestração, não a acurácia de um modelo;
- webcam, RTSP, GenICam, CUDA e latência real dependem do hardware final;
- PatchCore/EfficientAD exigem dataset, treino e critérios de aceitação;
- o dashboard local não substitui o futuro control plane SaaS multi-tenant.
