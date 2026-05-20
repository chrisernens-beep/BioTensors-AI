# BioTensors AI

Research preview of a **spiking neural network (SNN)** stack with STDP learning, layered cortical architecture, and optional multimodal inputs.

> **Status:** early research ([v0.1.0](https://github.com/chrisernens-beep/BioTensors-AI/releases/tag/v0.1.0)). The portable prototype uses ~1,100 neurons; larger-scale training runs on a separate workstation.

## Features

- **BioTensor** — LIF-style layer with adaptive thresholds, fatigue, homeostasis, and STDP
- **BioBrain** — sensory → associative → executive pipeline, plus limbic (safety) and Broca (speech) modules
- **BioBrainPro** — multimodal variant (health / navigation / speech primaries)
- **BioCompiler** — concept imprinting, ethical shielding, and weight export (`.bio`)

## Requirements

- Python 3.10+
- NumPy

```bash
pip install -r requirements.txt
```

## Quick start

Train the full curriculum and save weights:

```bash
python compile_biobrain.py
```

Fast demo (~1–5 s depending on hardware):

```bash
python reproduce_small_run.py
```

Optional GUI monitor (requires Tkinter, usually bundled with Python on Windows):

```bash
python multi_monitor.py
```

## Architecture (BioBrain)

| Layer       | Neurons | Role                          |
|------------|---------|-------------------------------|
| Sensory    | 400     | Spike encoding of input       |
| Associative| 400     | Integration + feedback loop   |
| Executive  | 100     | Action pools (4 decisions)    |
| Limbic     | 100     | Safety / ethical blocking     |
| Broca      | 100     | Speech sequencing             |

**Total:** 1,100 neurons · **Input size:** 100 spike dimensions

## Project layout

```
brain_architecture.py    # Core SNN layers and BioBrain
compile_biobrain.py      # Training / compilation pipeline
biobrainpro.py           # Multimodal BioBrainPro
bio_tokenizer.py         # Text → spike patterns
reproduce_small_run.py   # Minimal reproducible demo
multi_monitor.py         # Tkinter visualization (optional)
```

## Research direction

BioTensors targets **low-CPU, memory-centric autonomy** for constrained environments (e.g. onboard decision-making when ground contact is delayed). This repository is the public reference implementation of the portable prototype.

## Citation

If you use this work in research or a project, please cite:

```bibtex
@software{biotensors2026,
  author       = {Ernens, Christophe},
  title        = {BioTensors AI: Spiking Neural Network Brain with STDP Learning},
  year         = {2026},
  url          = {https://github.com/chrisernens-beep/BioTensors-AI},
  version      = {0.1.0},
  license      = {MIT}
}
```

A [CITATION.cff](CITATION.cff) file is also provided for GitHub's citation widget.

## Contact & collaboration

| | |
|---|---|
| **Author** | Christophe — pharmacist & independent developer (Belgium) |
| **Repository** | [github.com/chrisernens-beep/BioTensors-AI](https://github.com/chrisernens-beep/BioTensors-AI) |
| **Issues** | [Bug reports & ideas](https://github.com/chrisernens-beep/BioTensors-AI/issues) |
| **Collaboration** | Neuromorphic computing, onboard autonomy, robotics, space applications |

Open a [GitHub Issue](https://github.com/chrisernens-beep/BioTensors-AI/issues/new) to discuss experiments, integrations, or research partnerships.

## License

MIT — see [LICENSE](LICENSE).
