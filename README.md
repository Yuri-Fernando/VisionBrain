# 👁️ VisionBrain Engine

### Python · OpenCV · YOLO · GenICam · Edge AI · Event-Driven Vision · Streamlit

## Status

🟡 **MVP técnico funcional — núcleo local validado com 13 testes, lint limpo e notebook end-to-end executado integralmente.**

O fluxo sintético e as camadas determinísticas foram validados sem câmera, download de pesos ou serviço externo. Webcam, RTSP, GPU/CUDA e GenICam dependem do hardware, driver e runtime disponíveis no ambiente final e não são apresentados como já validados.

## Descrição / Contexto

Sistemas de visão computacional costumam começar pelo modelo: uma câmera entrega frames, um detector produz caixas e a demonstração termina ali. Em ambiente real, porém, muitos problemas aparecem antes e depois da inferência. Exposição inadequada, foco instável, clipping, ruído, latência acumulada, identidade temporal inconsistente e falhas na entrega de eventos podem tornar um modelo preciso pouco útil operacionalmente.

**VisionBrain** é um motor modular de visão computacional orientado à câmera e a eventos. Ele trata a formação da imagem, a qualidade da captura, o preprocessamento, a inferência, o tracking, as regras temporais e a persistência como partes do mesmo sistema — mantendo cada responsabilidade atrás de contratos substituíveis.

O projeto funciona localmente como laboratório de P&D, engine de edge computing e base para aplicações de inspeção visual e video analytics. Dashboard e notebook são camadas de apresentação sobre o mesmo pacote e os mesmos artefatos produzidos pelo CLI; não existem como implementações paralelas da lógica.

---

## 🌱 Como nasceu

O ponto de partida foi o **VisionGuard**, um experimento de detecção em tempo real que evitava processar ações redundantes: em vez de reagir a toda caixa detectada, ele mantinha estado e emitia uma ação somente quando surgia um evento relevante.

Esse experimento respondeu à primeira pergunta — *como transformar detecção contínua em eventos?* — mas abriu questões mais importantes:

- Como saber se a câmera realmente aplicou resolução, FPS ou exposição solicitados?
- Como distinguir falha de modelo de uma imagem escura, desfocada ou saturada?
- Como trocar webcam por RTSP ou GenICam sem reescrever o runtime?
- Como trocar o backend de inferência sem acoplar regras e eventos ao framework do modelo?
- Como manter a rede e o dashboard fora do loop crítico de frames?
- Como demonstrar o pipeline inteiro sem depender de webcam, GPU ou download de pesos?

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

## 🎯 Objetivos

- Caracterizar a fonte de imagem antes de executar inferência;
- Medir sinais operacionais de qualidade, como brilho, contraste, foco, clipping, ruído e color cast;
- Aplicar preprocessamento adaptativo somente quando os diagnósticos justificarem;
- Separar aquisição, inferência, tracking, regras, persistência e apresentação;
- Transformar detecções em eventos de negócio com snapshots e audit trail JSONL;
- Permitir demonstração e testes determinísticos sem hardware;
- Manter VLMs, webhooks e dashboards fora do caminho crítico de vídeo;
- Preparar o núcleo para edge computing, inspeção industrial e integração com um futuro control plane SaaS.

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

---

## 🧩 Módulos

| Módulo | Responsabilidade | Estado |
|---|---|---|
| VisionBrain Core | Configuração, modelos canônicos e runtime | ✅ Implementado |
| Camera Lab | Discovery, probe, autotune, diagnóstico e filtros | ✅ Implementado |
| Quality Gate | Brilho, contraste, foco, entropia, clipping, ruído e color cast | ✅ Implementado |
| Preprocess | Gamma, CLAHE, denoise e unsharp mask adaptativos | ✅ Implementado |
| Detection | YOLO com detecção e tracking persistente | ✅ Implementado · peso opcional |
| Analytics | Entrada/saída de zona, contagem e motion anomaly | ✅ Baseline implementada |
| Event Delivery | JSONL, snapshots, fila de webhook e spool de falhas | ✅ Implementado |
| Dashboard | KPIs, filtros, timeline, tabela e evidence viewer | ✅ Implementado |
| Notebook E2E | Tutorial completo usando fonte sintética | ✅ Executado localmente |
| GenICam | Adapter GenTL/Harvester integrado ao runtime | 🟡 Requer hardware + CTI |
| VLM/VQA | Perguntas sobre snapshots de eventos | 🟡 Adapter implementado |
| Learned Anomaly | PatchCore/EfficientAD e heatmaps | 🗺️ Roadmap |
| Edge Control Plane | API, identidade, deployment e rollback | 🗺️ Roadmap |

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

---

## 🚀 Quickstart sem webcam

Requer Python 3.11 ou superior.

```powershell
git clone https://github.com/Yuri-Fernando/VisionBrain.git
cd VisionBrain

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dashboard,notebook,dev]"
```

Execute o pipeline sintético completo:

```powershell
visionbrain demo --output outputs/demo --frames 24
```

Artefatos gerados:

```text
outputs/demo/
├── events.jsonl
├── summary.json
└── snapshots/
```

Abra o dashboard:

```powershell
visionbrain dashboard
```

No painel, selecione `outputs/demo/events.jsonl`.

O demo não acessa webcam, rede, GPU ou modelo externo. Ele valida aquisição, qualidade, preprocessamento, tracking simulado, regras, eventos, snapshots e contrato do dashboard.

---

## 📷 Teste com webcam e YOLO

```powershell
pip install -e ".[yolo,dev]"

visionbrain devices
visionbrain probe --source 0
visionbrain diagnose --source 0
visionbrain autotune --source 0
visionbrain filter-lab --source 0
visionbrain run --config config/fast_webcam.yaml --source 0
```

Controles da janela OpenCV:

```text
Q / ESC  encerrar
S        salvar snapshot manual
SPACE/P  pausar ou continuar
```

O backend e o driver podem aceitar uma propriedade sem aplicá-la exatamente. Sempre compare valores solicitados e observados antes de fixar uma receita de captura.

---

## 📊 Dashboard operacional

O dashboard é local e somente leitura. Ele não controla câmera nem executa inferência.

Recursos:

- total de eventos, tipos, severidades e evidências;
- filtro por tipo e severidade;
- distribuição de eventos;
- timeline agregada;
- tabela de auditoria;
- visualização dos snapshots associados.

Implementação: [`apps/dashboard/app.py`](apps/dashboard/app.py)
Guia operacional: [`DASHBOARD_AND_E2E.md`](DASHBOARD_AND_E2E.md)

---

## 📓 Notebook end-to-end

[`notebooks/visionbrain_end_to_end.ipynb`](notebooks/visionbrain_end_to_end.ipynb) percorre:

1. aquisição sintética;
2. análise de qualidade;
3. preprocessamento adaptativo;
4. detecção rastreada simulada;
5. regras de zona;
6. persistência JSONL;
7. geração e leitura de evidências;
8. contrato de dados do dashboard.

O notebook foi executado de cima a baixo e permanece salvo sem outputs embutidos para manter o repositório limpo.

---

## 🏭 GenICam / câmera industrial

Instale o extra:

```powershell
pip install -e ".[genicam]"
```

Configure o GenTL Producer fornecido pelo fabricante:

```yaml
camera:
  source: "genicam:C:/caminho/fabricante.cti"
```

O adapter participa do mesmo contrato `FrameSource`, mas uma implantação real ainda precisa validar pixel format, trigger, exposure, gain, packet size, iluminação, lente e estabilidade do driver no dispositivo final.

---

## 🔎 VQA sobre evidências

VLM não roda em cada frame. Ele é acionado somente sobre um snapshot selecionado:

```powershell
pip install -e ".[vlm]"

visionbrain vqa `
  --image outputs/snapshots/example.jpg `
  --question "Descreva qualquer anomalia visível nesta cena."
```

O modelo pode exigir download na primeira execução. A resposta multimodal é uma camada de explicação/triagem; decisões críticas continuam dependentes de regras e modelos validados para a tarefa.

---

## 🧪 Validação

Validação local realizada para a versão `0.2.0`:

```text
pytest:              13 passed
ruff:                All checks passed
compileall:          ok
CLI:                 comandos carregados
demo sintético:      executado
notebook E2E:        executado integralmente
```

Execute novamente:

```powershell
python -m pytest -q
python -m ruff check src tests apps
```

Os testes não abrem webcam, não baixam modelos e não chamam serviços externos.

---

## 📁 Estrutura

```text
VisionBrain/
├── apps/dashboard/              # interface Streamlit somente leitura
├── config/                      # receitas de runtime
├── docs/                        # arquitetura, pesquisa e integração SaaS
├── notebooks/                   # tutorial end-to-end
├── scripts/                     # atalhos Windows para webcam
├── src/visionbrain/
│   ├── anomaly/                 # baseline MOG2
│   ├── calibration/             # calibração intrínseca
│   ├── camera/                  # fontes, adapters e factory
│   ├── dashboard/               # loader e agregações do dashboard
│   ├── events/                  # regras, persistência e dispatcher
│   ├── inference/               # contratos, factory, YOLO e VLM
│   ├── preprocess/              # filtros e pipeline adaptativo
│   ├── quality/                 # métricas e recomendações
│   ├── benchmark.py
│   ├── demo.py
│   ├── runtime.py
│   └── cli.py
└── tests/
```

---

## 🧱 Evolução modular

O produto pode crescer sem concentrar todas as responsabilidades no mesmo processo:

```text
VisionBrain Core
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

### Próxima fase — analytics e runtime

- line crossing e contagem por trajetória;
- dwell time e occupancy;
- métricas de frame age e dropped frames;
- backend ONNX Runtime;
- captura/inferência com latest-frame buffer;
- structured logging e métricas Prometheus/OpenTelemetry.

### Inspeção industrial

- recipes fixas de câmera e iluminação;
- alinhamento de ROI e referência;
- PatchCore/EfficientAD;
- heatmaps e estados `PASS / REVIEW / FAIL`;
- fila de revisão humana e feedback para dataset.

### Edge / produto

- daemon FastAPI local;
- identidade de dispositivo;
- artefatos com checksum e assinatura;
- ativação atômica, shadow mode e rollback;
- control plane para câmeras, recipes e deployments.

Roadmap detalhado: [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## ⚠️ Limites e uso responsável

- O quality score é heurístico; thresholds devem ser calibrados por câmera e tarefa;
- MOG2 detecta mudança/movimento, não substitui anomaly detection industrial aprendida;
- A fonte sintética valida orquestração, não acurácia de modelo;
- Webcam, RTSP, CUDA e GenICam precisam de benchmark no hardware final;
- O dashboard atual é operacional/local, não multi-tenant;
- Ultralytics é opcional e possui licenciamento próprio. Uso comercial exige revisão das licenças de código, pesos, datasets e SDKs.

---

## 📚 Documentação

- [Arquitetura](docs/ARCHITECTURE.md)
- [Plano experimental](docs/EXPERIMENT_PLAN.md)
- [Notas de pesquisa](docs/RESEARCH_NOTES.md)
- [Integração SaaS](docs/SAAS_INTEGRATION.md)
- [Dashboard e E2E](DASHBOARD_AND_E2E.md)
- [Quickstart Windows](QUICKSTART.md)

---

## Licença

O core do VisionBrain é distribuído sob a licença Apache 2.0. Dependências e modelos opcionais mantêm suas próprias licenças e devem ser avaliados separadamente para cada cenário de distribuição.
