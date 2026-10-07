# Changelog

Todas as mudanças relevantes do VisionBrain são documentadas neste arquivo.

## [0.3.0] - 2026-10-07

### Adicionado

- detecção real de GPU/CUDA via torch (`visionbrain.inference.gpu_info.detect_gpu`),
  sem jamais lançar exceção quando torch ou CUDA estão ausentes;
- `recommend_detector_profile` em `camera/autotune.py`: autotune do device/resolução/
  precisão do detector com base na GPU real disponível (`cuda:0` + `imgsz=640` +
  `half=True` quando há CUDA; `cpu` + `imgsz=320` quando não há);
- `compare_cpu_gpu` em `benchmark.py`: mede latência/FPS reais do mesmo motor
  Ultralytics (mesmo modelo, mesmos frames sintéticos determinísticos) em CPU e em
  CUDA, sem estimativas — levanta `RuntimeError` se nenhuma GPU CUDA real for
  encontrada, em vez de reportar números de CPU como benchmark de GPU;
- comandos de CLI `visionbrain gpu-info` e `visionbrain benchmark-gpu`;
- benchmark real medido numa NVIDIA GeForce RTX 3060 Ti com o modelo já configurado
  (`yolo26n.pt`): ~4.36x de speedup CUDA fp16 vs CPU (28.6 FPS vs 6.56 FPS médios,
  120 frames por device) — ver `docs/benchmarks/gpu_rtx3060ti.md` e o JSON completo
  em `docs/benchmarks/gpu_rtx3060ti.json`;
- testes reais (não mockados) de detecção de GPU, autotune de device e comparação
  CPU/CUDA, com `pytest.mark.skipif` graceful quando não há GPU CUDA disponível.

### Alterado

- `benchmark.py` extraiu o cálculo de percentil para uma função de módulo
  reutilizável entre o benchmark síncrono existente e a nova comparação CPU/GPU.

### Contexto

Até a versão `0.2.0` o projeto validava apenas o caminho 100% sintético/sem hardware
(fonte determinística, sem câmera, sem GPU, sem download de modelo). Esta versão
adiciona uma validação real medida em hardware real (GPU NVIDIA disponível no
ambiente de desenvolvimento), sem remover o caminho sintético: os dois continuam
coexistindo, seguindo a mesma filosofia de "demonstração sem hardware + validação
real quando hardware existir" já documentada no README.

## [0.2.0] - 2026-10-03

### Adicionado

- contratos e factories para fontes de frame e backends de detecção;
- fonte sintética determinística para CI, demos e notebooks;
- adapter GenICam integrado ao runtime por `FrameSource`;
- dispatcher assíncrono de webhook com persistência local e spool;
- dashboard Streamlit para eventos e evidências;
- demo e notebook end-to-end sem hardware ou download de modelo;
- gravação de imagens compatível com caminhos Unicode no Windows;
- testes de arquitetura, runtime sintético, dashboard e fluxo E2E.

### Alterado

- runtime e benchmark deixaram de instanciar diretamente câmera e detector;
- documentação reorganizada para explicar origem, arquitetura modular, limites e evolução.

## [0.1.0] - 2026-10-03

- primeira fundação do engine: OpenCV, quality gate, preprocessamento, YOLO/tracking, eventos, benchmark, calibração, GenICam scaffold e VQA em snapshots.
