# Benchmark real: CPU vs GPU (RTX 3060 Ti)

Este arquivo documenta uma validação **real**, medida, não simulada: o mesmo motor de
inferência (`visionbrain.inference.yolo.YoloEngine`, Ultralytics) rodando o mesmo
modelo (`yolo26n.pt`, já configurado em `config/default.yaml`) contra os mesmos frames
determinísticos (`SyntheticCamera`), variando apenas o device de inferência.

## Ambiente medido

| Item | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti |
| Driver/CUDA (via torch) | CUDA 12.6 |
| torch | 2.13.0+cu126 |
| ultralytics | 8.4.24 |
| Modelo | `yolo26n.pt` (nano, peso baixado via Ultralytics assets) |
| Frames medidos por device | 120 (após 10 iterações de warm-up descartadas) |
| Resolução de inferência | `imgsz=640` nos dois devices (comparação justa) |
| Precisão | CPU em fp32; CUDA em fp16 (`half=True`) — perfil recomendado por `recommend_detector_profile` |
| Tracking | desabilitado durante a medição (isola custo de inferência pura, sem estado de tracker) |

## Resultado

| Device | Latência média | p50 | p95 | FPS médio |
|---|---|---|---|---|
| CPU | 152.41 ms | 135.70 ms | 235.61 ms | 6.56 |
| CUDA (fp16) | 34.97 ms | 32.33 ms | 48.43 ms | 28.60 |

**Speedup medido (CUDA vs CPU): ~4.36x.**

JSON completo (saída de `visionbrain benchmark-gpu`): [`gpu_rtx3060ti.json`](gpu_rtx3060ti.json).

## Como reproduzir

```powershell
pip install -e ".[yolo,dev]"
visionbrain gpu-info
visionbrain benchmark-gpu --config config/default.yaml --frames 120 --warmup-iters 10 --output docs/benchmarks/gpu_rtx3060ti.json
```

`visionbrain gpu-info` confirma `torch.cuda.is_available()` e o nome da GPU antes de
qualquer medição; `compare_cpu_gpu()` (`src/visionbrain/benchmark.py`) levanta
`RuntimeError` se nenhuma GPU CUDA real for encontrada, em vez de reportar números de
CPU como se fossem benchmark de GPU.

## O que isso valida e o que não valida

- Valida: o pipeline real de inferência (modelo real, pesos reais, device real) roda
  e produz speedup real nesta GPU específica, com números medidos (não estimados).
- Não valida: acurácia de detecção, comportamento em câmera real (webcam/RTSP/GenICam),
  nem desempenho em outra GPU — esses continuam dependendo do hardware/dataset do
  ambiente final, como já documentado no README.
- O caminho 100% sintético/sem hardware (`visionbrain demo`, `visionbrain benchmark`,
  notebook E2E) continua existindo e validado separadamente; esta GPU benchmark é um
  caminho adicional, não uma substituição.
