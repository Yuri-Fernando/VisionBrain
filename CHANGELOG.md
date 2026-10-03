# Changelog

Todas as mudanças relevantes do VisionBrain são documentadas neste arquivo.

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
