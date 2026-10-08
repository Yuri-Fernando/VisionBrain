# 👁️ VisionBrain Engine — motor modular de visão computacional orientado a eventos

### Python · OpenCV · YOLO (Ultralytics) · PyTorch/CUDA · GenICam · Streamlit · Edge AI

[![ci](https://github.com/Yuri-Fernando/VisionBrain/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuri-Fernando/VisionBrain/actions/workflows/ci.yml)
![testes](https://img.shields.io/badge/testes-18%20passed%20%2B%201%20skip-0a8a0a)
![GPU](https://img.shields.io/badge/GPU%20real-RTX%203060%20Ti%20~4.36x-c8742a)
![versão](https://img.shields.io/badge/vers%C3%A3o-v0.3.0-4a5563) ![license](https://img.shields.io/badge/license-Apache%202.0-4a5563)

## Status

🟡 **MVP técnico funcional — núcleo local validado nesta revisão: 18 testes passaram e 1 foi pulado (skip esperado, máquina sem GPU CUDA), lint limpo e notebook end-to-end já executado integralmente em sessão anterior.**

O fluxo sintético e as camadas determinísticas (quality gate, preprocessamento, tracking simulado, eventos, dashboard) são validados sem câmera, download de pesos ou serviço externo. Webcam, RTSP e GenICam continuam dependendo do hardware, driver e runtime do ambiente final e não são apresentados como já validados.

GPU/CUDA, por outro lado, **já foi validada com hardware real** em sessão anterior: numa NVIDIA GeForce RTX 3060 Ti, o mesmo motor Ultralytics com o mesmo modelo (`yolo26n.pt`) mediu CPU 152,41 ms/6,56 FPS vs CUDA fp16 34,97 ms/28,60 FPS — **speedup medido de ~4,36x**. Números completos em [`docs/benchmarks/gpu_rtx3060ti.md`](docs/benchmarks/gpu_rtx3060ti.md). Isso valida o device de inferência nessa GPU específica — não valida webcam/RTSP/GenICam nem desempenho em outra GPU. O teste correspondente (`tests/test_gpu_benchmark.py`) é pulado automaticamente em máquina sem CUDA, como ocorreu nesta revisão; o caminho 100% sintético/sem hardware continua existindo e suportado para quem não tiver GPU.

---

## Descrição / Contexto

Sistemas de visão computacional costumam começar pelo modelo: uma câmera entrega frames, um detector produz caixas e a demonstração termina ali. Em ambiente real, porém, muitos problemas aparecem antes e depois da inferência. Exposição inadequada, foco instável, clipping, ruído, latência acumulada, identidade temporal inconsistente e falhas na entrega de eventos podem tornar um modelo preciso pouco útil operacionalmente.

**VisionBrain** é um motor modular de visão computacional orientado à câmera e a eventos. Ele trata a formação da imagem, a qualidade da captura, o preprocessamento, a inferência, o tracking, as regras temporais e a persistência como partes do mesmo sistema — mantendo cada responsabilidade atrás de contratos substituíveis.

O projeto funciona localmente como laboratório de P&D, engine de edge computing e base para aplicações de inspeção visual e video analytics. Dashboard e notebook são camadas de apresentação sobre o mesmo pacote e os mesmos artefatos produzidos pelo CLI; não existem como implementações paralelas da lógica.

---

## 🌱 Como nasceu

O ponto de partida foi o **VisionGuard**, um experimento de detecção em tempo real que evitava processar ações redundantes: em vez de reagir a toda caixa detectada, ele mantinha estado e emitia uma ação somente quando surgia um evento relevante.

Esse experimento respondeu à primeira pergunta — *como transformar detecção contínua em eventos?* — mas abriu questões mais importantes:

1. Como saber se a câmera realmente aplicou resolução, FPS ou exposição solicitados?
2. Como distinguir falha de modelo de uma imagem escura, desfocada ou saturada?
3. Como trocar webcam por RTSP ou GenICam sem reescrever o runtime?
4. Como trocar o backend de inferência sem acoplar regras e eventos ao framework do modelo?
5. Como manter a rede e o dashboard fora do loop crítico de frames?
6. Como demonstrar o pipeline inteiro sem depender de webcam, GPU ou download de pesos?

O VisionBrain nasceu dessa mudança de perspectiva. A câmera deixou de ser apenas uma entrada e passou a ser uma fonte caracterizada; o frame ganhou métricas de qualidade; as detecções ganharam identidade temporal; e as saídas passaram a representar eventos auditáveis com evidência.

```text
VisionGuard
detecção -> comparação de estado -> evento

                         evoluiu para

VisionBrain
sensor -> aquisição -> quality gate -> preprocessamento -> inferência
       -> tracking -> regras temporais -> eventos -> evidências -> integrações
```

---

## 🎯 Objetivo

- Caracterizar a fonte de imagem antes de executar inferência;
- Medir sinais operacionais de qualidade, como brilho, contraste, foco, clipping, ruído e color cast;
- Aplicar preprocessamento adaptativo somente quando os diagnósticos justificarem;
- Separar aquisição, inferência, tracking, regras, persistência e apresentação;
- Transformar detecções em eventos de negócio com snapshots e audit trail JSONL;
- Permitir demonstração e testes determinísticos sem hardware;
- Validar o caminho de GPU real quando houver hardware disponível, sem tornar isso pré-requisito;
- Manter VLMs, webhooks e dashboards fora do caminho crítico de vídeo;
- Preparar o núcleo para edge computing, inspeção industrial e integração com um futuro control plane SaaS.

---

## 🔬 Linha de Pesquisa / Desenvolvimento

- Diagnóstico óptico e quality gate de captura (brilho, contraste, foco, entropia, clipping, ruído, color cast);
- Preprocessamento adaptativo (gamma, CLAHE, denoise, unsharp mask) disparado por diagnóstico, não por padrão fixo;
- Inferência YOLO via Ultralytics com tracking persistente, desacoplada por protocolo (`DetectorBackend`);
- Autotune de device/resolução/precisão (`cuda:0` + `imgsz=640` + `half=True` quando há CUDA; `cpu` + `imgsz=320` quando não há);
- Benchmark real CPU vs GPU sobre o mesmo motor e os mesmos frames determinísticos;
- Regras temporais/zonas para transformar detecção contínua em eventos auditáveis;
- VQA sobre evidências (snapshot, não cada frame) como camada de explicação opcional.

---

## 🏗️ Arquitetura

```text
                     FONTES DE IMAGEM
       webcam / vídeo / RTSP      GenICam       sintética
                 │                   │               │
                 └──────── FrameSource Protocol ─────┘
                                     │
                                     ▼
                      capability probe / warm-up
                                     │
                                     ▼
                    quality gate + diagnóstico óptico
                                     │
                                     ▼
                preprocessamento adaptativo por receita
                                     │
                                     ▼
                         DetectorBackend Protocol
                     Ultralytics hoje · novos backends
                                     │
                                     ▼
                       tracking + motion baseline
                                     │
                                     ▼
                    zonas + contagem + regras temporais
                                     │
                                     ▼
                                  eventos
                         ┌───────────┼───────────┐
                         ▼           ▼           ▼
                       JSONL      snapshots    fila assíncrona
                                                   │
                                                   ▼
                                             webhook / SaaS

        Dashboard Streamlit e notebook leem os mesmos artefatos locais
```

### Princípios de projeto

1. **Sensor antes do modelo** — iluminação, exposição, óptica e foco fazem parte do problema;
2. **Core desacoplado** — fontes e detectores são selecionados por factories e protocolos;
3. **Loop determinístico** — rede, VLM e interface não bloqueiam a captura/inferência;
4. **Eventos, não flood de frames** — persistir semântica e evidência relevante;
5. **Offline-first para desenvolvimento** — fonte sintética reproduzível e sem dependências externas;
6. **Limites explícitos** — integração disponível não significa hardware validado.

### Módulos

| Módulo | Função | Teste/Evidência |
|---|---|---|
| VisionBrain Core | Configuração, modelos canônicos e runtime | Testes de arquitetura/runtime |
| Camera Lab | Discovery, probe, autotune, diagnóstico e filtros | `visionbrain probe/diagnose/autotune` |
| Quality Gate | Brilho, contraste, foco, entropia, clipping, ruído, color cast | Testes de qualidade sintética |
| Preprocess | Gamma, CLAHE, denoise e unsharp mask adaptativos | Testes do pipeline adaptativo |
| Detection | YOLO (Ultralytics) com tracking persistente | GPU validada em hardware real (RTX 3060 Ti) |
| Analytics | Entrada/saída de zona, contagem e motion anomaly (MOG2) | Baseline implementada e testada |
| Event Delivery | JSONL, snapshots, fila de webhook e spool de falhas | Testes de persistência/evento |
| Dashboard | KPIs, filtros, timeline, tabela e evidence viewer | `apps/dashboard/app.py`, somente leitura |
| Notebook E2E | Tutorial completo usando fonte sintética | Executado integralmente (sessão anterior) |
| GenICam | Adapter GenTL/Harvester integrado ao runtime | 🟡 Requer hardware + CTI, não validado |
| VLM/VQA | Perguntas sobre snapshots de eventos | 🟡 Adapter implementado, download sob demanda |

---

## ⚙️ Funcionamento

1. A factory cria uma fonte OpenCV, GenICam ou sintética a partir da configuração;
2. A fonte é aquecida e seus dados efetivos são registrados por `probe()`;
3. O quality gate amostra frames e calcula métricas individuais de captura;
4. O pipeline aplica apenas as correções habilitadas e justificadas pelos diagnósticos;
5. O backend de inferência produz detecções normalizadas, sem expor o framework às regras;
6. O tracking mantém identidade e permite reconhecer entrada, saída e permanência lógica;
7. O Event Engine avalia zonas, contagens, qualidade degradada e anomalia de movimento;
8. Cada evento é persistido primeiro em JSONL, recebe uma evidência visual quando configurado e pode ser encaminhado por webhook fora do loop crítico;
9. Dashboard, notebook e integrações consomem o mesmo contrato de evento.

```text
fonte → probe/warm-up → quality gate → preprocess adaptativo → detecção (CPU/GPU) →
tracking → zonas/regras → evento → JSONL + snapshot → webhook (opcional) → dashboard/notebook
```

---

## 🧠 Verificação / Validação

| Camada | O que prova | Resultado |
|---|---|---|
| Suíte sintética (`pytest`) | Orquestração determinística sem hardware | 18 passed, 1 skipped nesta revisão (máquina sem CUDA) |
| `ruff check` | Estilo e erros estáticos | Limpo na última verificação registrada |
| Notebook E2E | Pipeline completo ponta a ponta sem hardware | Executado integralmente, salvo sem outputs embutidos |
| `visionbrain gpu-info` + `benchmark-gpu` | Device real, não estimado | `torch.cuda.is_available()` confirmado em RTX 3060 Ti; `RuntimeError` se não houver CUDA real |
| `tests/test_gpu_benchmark.py` | Benchmark real CPU vs GPU no mesmo motor | 1 teste de guard roda sempre; o teste de medição real roda só com CUDA disponível (`skipif`) |

O oráculo de qualidade (quality gate) é heurístico e documentado como tal — ver Limitações.

---

## 🧪 Desenvolvimento Experimental

**Achado real (não plantado):** até a versão `0.2.0`, o projeto só validava o caminho 100% sintético/sem hardware. A pergunta era se o mesmo motor de inferência (Ultralytics) realmente usava a GPU quando disponível, ou se "suporte a GPU" era apenas uma opção de configuração nunca medida.

- **Hipótese:** o device `cuda:0` reduz a latência de inferência de forma mensurável frente a `cpu`, sobre o mesmo modelo e os mesmos frames.
- **O que se testou:** `compare_cpu_gpu()` roda o mesmo `yolo26n.pt` nos dois devices, sobre os mesmos 120 frames sintéticos determinísticos (após warm-up), medindo latência/FPS reais — sem estimativa.
- **Causa/resultado real:** numa RTX 3060 Ti real, CPU mediu 152,41 ms/6,56 FPS e CUDA fp16 mediu 34,97 ms/28,60 FPS — speedup de ~4,36x. A função levanta `RuntimeError` se chamada sem GPU CUDA real, para nunca reportar número de CPU como se fosse benchmark de GPU.
- **Correção/decisão de design:** o teste que mede isso de verdade (`test_compare_cpu_gpu_measures_real_latency_on_available_hardware`) é marcado `skipif` quando `torch.cuda.is_available()` é `False` — confirmado nesta revisão (18 passed, 1 skipped), em vez de falhar ou mockar GPU inexistente.

Detalhes e ambiente completo: [`docs/benchmarks/gpu_rtx3060ti.md`](docs/benchmarks/gpu_rtx3060ti.md).

---

## 🛠️ Tecnologias

- **Linguagem:** Python 3.11+
- **Visão computacional:** OpenCV, Ultralytics (YOLO)
- **GPU/ML runtime:** PyTorch + CUDA (quando disponível)
- **Câmera industrial:** GenICam / GenTL (Harvester), opcional
- **Apresentação:** Streamlit (dashboard), Jupyter (notebook E2E)
- **CLI:** `visionbrain` (Typer/Click-style)
- **Qualidade/CI:** pytest, ruff, GitHub Actions

---

## 📊 Resultados

| Métrica | Valor | Origem |
|---|---|---|
| Testes (nesta revisão, sem GPU) | 18 passed, 1 skipped | `python -m pytest -q`, executado nesta sessão |
| Latência CPU (YOLO, `yolo26n.pt`) | 152,41 ms / 6,56 FPS médios | `docs/benchmarks/gpu_rtx3060ti.md`, medido em sessão anterior na RTX 3060 Ti |
| Latência CUDA fp16 (mesmo modelo) | 34,97 ms / 28,60 FPS médios | idem |
| Speedup GPU vs CPU | ~4,36x | calculado a partir dos dois valores acima |
| Notebook E2E | Executado integralmente | `notebooks/visionbrain_end_to_end.ipynb`, salvo sem outputs |

O benchmark de GPU é válido para esta GPU específica (RTX 3060 Ti) e este modelo (`yolo26n.pt`); não generaliza para outro hardware sem nova medição.

---

## 🚀 Aplicações

- Laboratório de P&D para pipelines de visão orientados a eventos;
- Base para inspeção visual industrial (quality gate + futura anomaly detection aprendida);
- Video analytics local (contagem, zonas, dwell time) sem depender de serviço externo;
- Prototipação de edge AI antes de integrar a um control plane/SaaS.

---

## 🔭 Visão de Longo Prazo

```text
VisionBrain Core (hoje)
   ├── Camera Lab       diagnóstico, calibração e receitas
   ├── Inspect          anomalia, segmentação e PASS/REVIEW/FAIL
   ├── Analytics        zonas, linhas, dwell time e trajetórias
   ├── Multimodal       VQA e explicação de evidências
   └── Edge Agent       identidade, spool, health e deployments
                              │
                              ▼
                     Control Plane / SaaS
              fleet · recipes · deployments · review
```

O edge permanece responsável por captura, inferência e eventos de baixa latência. Um futuro SaaS gerencia identidade, frota, versões, histórico, revisão humana e deployments — recebendo eventos e evidências selecionadas em vez de vídeo bruto contínuo.

---

## 🗺️ Roadmap

- **F0 — Fundação do engine** ✅ Concluída — OpenCV, quality gate, preprocessamento, YOLO/tracking, eventos, benchmark, calibração, GenICam scaffold, VQA em snapshots (`v0.1.0`).
- **F1 — Arquitetura modular e dashboard** ✅ Concluída — contratos/factories, fonte sintética determinística, dispatcher de webhook, dashboard Streamlit, demo e notebook E2E sem hardware (`v0.2.0`).
- **F2 — Validação de GPU real** ✅ Concluída — detecção real de CUDA, autotune de device, benchmark CPU vs GPU medido em RTX 3060 Ti, ~4,36x de speedup (`v0.3.0`).
- **F3 — Analytics e runtime** ⏳ Planejada — line crossing, dwell time, occupancy, frame age/dropped frames, backend ONNX Runtime, structured logging/Prometheus.
- **F4 — Inspeção industrial** ⏳ Planejada — recipes fixas de câmera/iluminação, PatchCore/EfficientAD, heatmaps, fila de revisão humana.
- **F5 — Edge / produto** ⏳ Planejada — daemon FastAPI local, identidade de dispositivo, deployments assinados, control plane.

Roadmap detalhado: [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## 🕓 Histórico e Mudanças

| Versão | Data | O que mudou |
|---|---|---|
| 0.3.0 | 2026-10-07 | Detecção real de GPU/CUDA, autotune de device/resolução/precisão, benchmark real CPU vs GPU (RTX 3060 Ti, ~4,36x), CLI `gpu-info`/`benchmark-gpu`, testes reais com `skipif` gracioso |
| 0.2.0 | 2026-10-03 | Contratos/factories de fonte e detector, fonte sintética determinística, adapter GenICam, dispatcher de webhook, dashboard Streamlit, demo/notebook E2E |
| 0.1.0 | 2026-10-03 | Fundação do engine: OpenCV, quality gate, preprocessamento, YOLO/tracking, eventos, benchmark, calibração, GenICam scaffold, VQA |

Changelog completo: [`CHANGELOG.md`](CHANGELOG.md).

---

## 🔮 Próximos Passos

**Concluído nesta versão (0.3.0):**
- ✅ Detecção real de GPU/CUDA e autotune de device;
- ✅ Benchmark real CPU vs GPU medido em hardware (RTX 3060 Ti).

**Dependem de hardware/terceiros:**
- Validação de desempenho em outra GPU (cada GPU precisa de medição própria);
- Validação de webcam/RTSP/GenICam no hardware/driver final de produção.

**Próximos técnicos (o que faria numa v2):**
- Line crossing, dwell time e occupancy sobre o mesmo tracking;
- Backend ONNX Runtime como alternativa ao Ultralytics;
- Structured logging e métricas Prometheus/OpenTelemetry;
- PatchCore/EfficientAD para anomaly detection aprendida.

---

## 🚀 Como rodar localmente

Pré-requisito: Python 3.11+.

```powershell
git clone https://github.com/Yuri-Fernando/VisionBrain.git
cd VisionBrain

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dashboard,notebook,dev]"
```

Pipeline sintético completo (sem webcam/GPU/rede):

```powershell
visionbrain demo --output outputs/demo --frames 24
visionbrain dashboard
```

Com webcam e YOLO:

```powershell
pip install -e ".[yolo,dev]"
visionbrain devices
visionbrain probe --source 0
visionbrain run --config config/fast_webcam.yaml --source 0
```

Benchmark real CPU vs GPU (requer GPU CUDA):

```powershell
pip install -e ".[yolo,dev]"
visionbrain gpu-info
visionbrain benchmark-gpu --config config/default.yaml --frames 120 --warmup-iters 10 --output docs/benchmarks/gpu_rtx3060ti.json
```

Testes e lint (executados nesta revisão):

```powershell
python -m pytest -q        # 18 passed, 1 skipped nesta máquina (sem CUDA)
python -m ruff check src tests apps
```

| Ambiente | Observação |
|---|---|
| Windows (PowerShell) | Caminho principal de desenvolvimento e dos scripts em `scripts/` |
| Sem GPU | Caminho sintético/webcam completo; teste de benchmark GPU é pulado |
| Com GPU CUDA | Habilita `gpu-info`/`benchmark-gpu` e o teste de medição real |
| CI (GitHub Actions) | `.github/workflows/ci.yml`, roda a suíte sem hardware |

---

## 📁 Estrutura do repositório

```text
VisionBrain/
├── apps/dashboard/              # interface Streamlit somente leitura
├── config/                      # receitas de runtime
├── docs/                        # arquitetura, pesquisa, benchmarks e integração SaaS
│   └── benchmarks/gpu_rtx3060ti.md   # benchmark real CPU vs GPU
├── notebooks/                   # tutorial end-to-end
├── scripts/                     # atalhos Windows para webcam
├── src/visionbrain/
│   ├── anomaly/                 # baseline MOG2
│   ├── calibration/             # calibração intrínseca
│   ├── camera/                  # fontes, adapters, factory e autotune
│   ├── dashboard/               # loader e agregações do dashboard
│   ├── events/                  # regras, persistência e dispatcher
│   ├── inference/               # contratos, factory, YOLO, gpu_info e VLM
│   ├── preprocess/               # filtros e pipeline adaptativo
│   ├── quality/                  # métricas e recomendações
│   ├── benchmark.py              # benchmark síncrono + compare_cpu_gpu
│   ├── demo.py
│   ├── runtime.py
│   └── cli.py
└── tests/
```

---

## 📚 Documentação

| Documento | Conteúdo |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Arquitetura detalhada |
| [docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) | Plano experimental |
| [docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md) | Notas de pesquisa |
| [docs/SAAS_INTEGRATION.md](docs/SAAS_INTEGRATION.md) | Integração com futuro SaaS |
| [DASHBOARD_AND_E2E.md](DASHBOARD_AND_E2E.md) | Guia operacional do dashboard e E2E |
| [QUICKSTART.md](QUICKSTART.md) | Quickstart Windows |
| [docs/benchmarks/gpu_rtx3060ti.md](docs/benchmarks/gpu_rtx3060ti.md) | Benchmark real CPU vs GPU (RTX 3060 Ti) |
| [CHANGELOG.md](CHANGELOG.md) | Histórico completo de versões |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Roadmap detalhado |

---

## ⚠️ Limitações

- O quality score é heurístico; thresholds devem ser calibrados por câmera e tarefa;
- MOG2 detecta mudança/movimento, não substitui anomaly detection industrial aprendida;
- A fonte sintética valida orquestração, não acurácia de modelo;
- Webcam, RTSP e GenICam precisam de benchmark no hardware final; CUDA já foi benchmarkada em hardware real (RTX 3060 Ti), mas uma GPU diferente deve ser medida novamente antes de qualquer decisão de capacidade — não verificado nesta revisão para outro hardware;
- O dashboard atual é operacional/local, não multi-tenant;
- Ultralytics é opcional e possui licenciamento próprio; uso comercial exige revisão das licenças de código, pesos, datasets e SDKs;
- Esta revisão rodou a suíte numa máquina sem GPU CUDA — o teste de medição real de GPU ficou `skipped`, como esperado, e não foi reexecutado contra hardware nesta sessão.

---

## Status

🟡 **MVP técnico funcional.** Núcleo sintético e determinístico validado (18 passed, 1 skipped nesta revisão), notebook E2E executado integralmente, e GPU real validada em sessão anterior numa RTX 3060 Ti (~4,36x de speedup medido). Webcam, RTSP e GenICam seguem dependentes de hardware/driver final e não estão validados neste repositório.

---

## Contexto / Observações

Projeto de portfólio pessoal, não afiliado a nenhuma empresa ou cliente. Não há deploy em produção, usuários reais ou escala além do ambiente de desenvolvimento descrito aqui. Os números de GPU são medições reais de uma execução específica (`visionbrain benchmark-gpu`) em uma RTX 3060 Ti; não são estimativas de ferramenta nem projeções, e não devem ser generalizados para outro hardware sem nova medição.

---

## 🔗 Projetos Relacionados

| Projeto | Relação |
|---|---|
| VisionGuard | Predecessor direto — experimento de detecção→evento que motivou a arquitetura modular do VisionBrain |

---

## 🤖 Autor

**Yuri Fernando Dubbern**

Engenharia Elétrica · Ciência da Computação · Inteligência Artificial ·
Sistemas Embarcados · Projeto de Hardware · Pesquisa e Desenvolvimento

[LinkedIn](https://www.linkedin.com/in/yuridubbern) · [GitHub](https://github.com/Yuri-Fernando) · [Lattes](http://lattes.cnpq.br/7151392692642166) · [Linktree](https://linktr.ee/yuri.f.dubbern)

## Licença

Apache License 2.0 — ver [LICENSE](LICENSE). Dependências e modelos opcionais (ex.: Ultralytics) mantêm suas próprias licenças e devem ser avaliadas separadamente para cada cenário de distribuição.
