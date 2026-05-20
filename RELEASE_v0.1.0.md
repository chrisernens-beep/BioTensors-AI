# BioTensors AI v0.1.0 — Research Preview

First public release of the BioTensors SNN prototype.

## Highlights

- **BioTensor** layer with STDP, adaptive thresholds, fatigue, and homeostasis
- **BioBrain** architecture (~1,100 neurons): sensory → associative → executive + limbic + Broca
- **BioBrainPro** multimodal variant (health / navigation / speech)
- **BioCompiler** training pipeline with concept imprinting and ethical shielding
- Reproducible demo via `reproduce_small_run.py`

## Quick start

```bash
pip install -r requirements.txt
python reproduce_small_run.py
```

## What's next

- Larger-scale training (100k+ neurons) on workstation
- Mission-style demo (degraded ground link, sensor events)
- Broca / vocabulary dimension alignment fix

## Full changelog

See commit history: https://github.com/chrisernens-beep/BioTensors-AI/commits/main
