# Adversarial Deepfake Generation and Purification-Based Adversarial Detection
### Team *Go To Germany* — ImageCLEF 2026 Deepfake Task @ CLEF 2026

[![Paper](https://img.shields.io/badge/paper-CEUR--WS-blue)](paper/imageclef_camera_ready.pdf)
[![Slides](https://img.shields.io/badge/slides-PDF-orange)](presentation/)
[![Leaderboard](https://img.shields.io/badge/official-leaderboard-red)](https://www.imageclef.org/2026/deepfake-detection-and-generation)
[![Conference](https://img.shields.io/badge/CLEF-2026-green)](https://clef2026.clef-initiative.eu/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

> One attacker–defender exchange across three tracks: a transferable adversarial generation pipeline, a complementary dual-specialist detection ensemble, and a weight-free purification signal for flagging adversarial inputs.

Presented at **CLEF 2026**, Friedrich-Schiller-Universität Jena, Germany (Sep 24, 2026).
Official task page and final leaderboard: **[imageclef.org/2026/deepfake-detection-and-generation](https://www.imageclef.org/2026/deepfake-detection-and-generation)**

---

## Overview

This repository contains the code, paper, and presentation materials for our submission to the ImageCLEF 2026 Deepfake Detection and Generation Task. The work is organized along three tracks:

| Track | Approach | Final Score / Key Result |
|---|---|---|
| **Generation** | FLUX.1-dev + ControlNet + PuLID + custom 12-detector PGD (DiffJPEG-in-loop, MI/DI/EoT) | **0.4170** (90% evasion on organizer, 58% on participant detectors) |
| **Detection** | Two-specialist ensemble (SigLIP+DINOv2 × GenD-DINOv3) with max-probability rule and τ_min calibration | **0.6986** (99.4% baseline-fake, 82% participant-fake accuracy) |
| **Further Experiments** | Input-space purification: raw \|Δlogit\| under median-3 through EFFORT | **AUROC 0.81–0.98** across four adversarial source types |

> Full participant rankings for both sub-tasks: **[ImageCLEF 2026 Deepfake official leaderboard →](https://www.imageclef.org/2026/deepfake-detection-and-generation)**

Beyond the two official sub-tasks, we conducted a self-initiated investigation of purification-based adversarial detection, comparing three signal families across six detectors sharing a CLIP ViT-L/14 backbone. The strongest signal — raw \|Δlogit\| under median-3 purification through the EFFORT detector — refutes a simple backbone-preservation hypothesis and exposes a sharp **JPEG-quality cliff at Q70** where the signal collapses.

---

## Pipeline

### Generation
```
Reference face crops ──► PuLID ──┐
                                 ├──► FLUX.1-dev (DiT) ──► clean fake ──► PGD × 25 ──► adversarial image
MediaPipe FaceMesh → OpenPose ──► ControlNet ──┘                            │
                                                                            └── 12 detectors (simultaneous)
                                                                                DiffJPEG (ε=2/255, Q=80 in-loop)
                                                                                MI (μ=1.0) + DI (p=0.5) + EoT
```

### Detection
```
Input ──► pHash short-circuit ──► Fake?
   │             │
   │           No match
   ▼
SigLIP+DINOv2 ─── p_ai
GenD-DINOv3   ─── p_fm
         │
   max(p_ai, p_fm) > 0.5  OR  min(p_ai, p_fm) > τ_min  →  Fake
                                          else           →  Real
```

### Further Experiments (Purification)
```
Input x ──┐
          ├──► EFFORT (CLIP ViT-L/14 + SVD residual) ──► logit f(x)
          │                                                 │
          └── median-3 ──► T(x) ──► EFFORT ──► logit f(T(x))
                                                            │
                                               |Δlogit| = |f(T(x)) − f(x)|
                                                            │
                                                      Adversarial if large
```

---

## Repository Structure

```
clef2026-deepfake-adversarial/
├── paper/                             # Camera-ready paper (CEUR-WS format)
├── presentation/                      # Conference slides + Q&A preparation notes
├── generation/
│   ├── pipeline/                      # FLUX + ControlNet + PuLID
│   ├── attack/                        # 12-detector PGD + MI/DI/EoT + DiffJPEG
│   ├── evaluation/                    # Internal validation + transferability
│   └── preprocessing/                 # Face extraction utilities
├── detection/
│   ├── phash_shortcircuit/            # Perceptual-hash exact match
│   └── submission_logic/              # Ensemble inference entrypoint
└── further_experiments/
    ├── instability/                   # §6.1 — multi-detector cross-crop (failed, JPEG confound)
    ├── mahalanobis/                   # §6.2 — DINOv2 feature-space outlier (failed, distribution shift)
    ├── jpeg_confound/                 # Characterization of the JPEG confound
    ├── adv_vs_jpeg/                   # Multi-operator LR on single detector
    ├── finetuning_ablation/           # §6.3.3 — 6-detector purification sweep (partial — see note below)
    ├── purification/                  # Supporting purification experiments
    └── epsilon_ablation/              # PGD ε sensitivity sweep
```

> **Note:** The final fine-tuning ablation experiment (6-detector CLIP ViT-L/14 comparison, EFFORT median-3 sweep across JPEG qualities) was conducted on separate compute and the full training scripts are maintained in a private Google Drive location. The scripts included here cover the inference and purification portions. Full training code available on request.

---

## Key Results

### Generation (Table 1 in paper)

| Target | Evasion rate |
|---|---|
| 12 white-box targets (internal) | ≥95% on 15/17 detectors |
| Organizer detector pool | **90.19%** |
| Participant detector pool | 57.62% |
| **Final generation score** | **0.4170** |

### Detection (§4 in paper)

| Metric | Value |
|---|---|
| Baseline deepfake accuracy | 99.39% |
| Participant deepfake accuracy | 88.22% |
| Real-side accuracy | 43–57% (calibration mistake) |
| **Final detection score** | **0.6986** |

### Further Experiments — EFFORT + median-3 AUROC (Table 12)

| Source | Raw input | JPEG-Q80 |
|---|---|---|
| Production (own PGD, DiffJPEG-Q80 in-loop) | **0.982** | 0.789 |
| Cross AI-generated (BitMind SDXL/FLUX, PGD+FGSM+BIM) | **0.808** | 0.580 |
| Cross face-swap (FF++ Deepfakes/FaceShifter/FaceSwap) | **0.891** | 0.681 |
| Cross reenactment (FF++ Face2Face/NeuralTextures) | **0.890** | 0.722 |

**JPEG-quality cliff:** signal collapses at Q70 (all sources drop to AUROC ~0.47 random), recovers to ≥0.85 at Q95.

---

## Technologies

- **Models:** FLUX.1-dev, PuLID, ControlNet-Union-Pro-2.0, SigLIP+DINOv2, GenD-DINOv3, EFFORT (CLIP ViT-L/14), ForAda
- **Frameworks:** PyTorch, HuggingFace Diffusers, HuggingFace Transformers
- **Attacks:** Custom PGD with DiffJPEG, Momentum Iterative (MI), Input Diversity (DI), implicit Expectation over Transformation (EoT)
- **Preprocessing:** MediaPipe FaceMesh → OpenPose conversion
- **Analysis:** DeepfakeBench baseline detectors, scikit-learn (logistic regression ensemble)

---

## Reproduction Notes

This repository is organized for study and portfolio review rather than one-click reproduction — model weights, datasets, and HuggingFace tokens are intentionally excluded (see `.gitignore`). To reproduce:

1. **Install dependencies** (see `requirements.txt` and `generation/requirements.txt`)
2. **Set environment variables** — `HF_TOKEN` for FLUX.1-dev and PuLID access
3. **Download pretrained weights** — FLUX.1-dev, PuLID, ControlNet-Union-Pro-2.0, EFFORT, SigLIP+DINOv2, GenD-DINOv3 from HuggingFace
4. **Prepare data** — ImageCLEF 2026 Deepfake Development Videos (requires organizer access)
5. **Generation entry point:** `generation/pipeline/generate_1002.py` → `generation/attack/pgd_12detector.py`
6. **Detection entry point:** `detection/submission_logic/run_all.py`
7. **Further experiments:** See individual subdirectories under `further_experiments/`

All hardcoded paths beginning with `<REPO_ROOT>` were previously absolute paths to the development environment; set them to your local checkout root.

---

## Paper & Presentation

- 📄 **Paper (CEUR-WS):** [`paper/imageclef_camera_ready.pdf`](paper/imageclef_camera_ready.pdf)
- 🎤 **Slides:** [`presentation/slides.pptx`](presentation/slides.pptx)
- ❓ **Q&A preparation (PGD, DI, MI, EoT, DiffJPEG, Mahalanobis rescue attempts):** [`presentation/qna_preparation.pdf`](presentation/qna_preparation.pdf)

---

## Citation

```bibtex
@inproceedings{Kim2026ImageCLEFDeepfake,
  author    = {Junghyun Kim and Seunghyun Kim and Jiyoung Woo},
  title     = {Adversarial Deepfake Generation and an Investigation of Purification-Based Adversarial Detection: Notebook for the ImageCLEF at CLEF 2026},
  booktitle = {CLEF 2026 Working Notes},
  series    = {CEUR Workshop Proceedings},
  year      = {2026},
  address   = {Jena, Germany},
  publisher = {CEUR-WS.org}
}
```

---

## Authors

- **Junghyun Kim** (First author, methodology & implementation) — [GitHub](https://github.com/jhyeun1020) · jhyeun1020@gmail.com
- **Seunghyun Kim** (Cross-track discussion & manuscript review)
- **Jiyoung Woo** (Supervisory feedback) — Department of AI and Big Data Engineering, Soonchunhyang University

## Acknowledgments

This research was supported by the Ministry of Science and ICT (MSIT), Korea and the Institute of Information & Communications Technology Planning & Evaluation (IITP) under the AI University program (2026-0-00032, 2026).

## License

Code released under the [MIT License](LICENSE). The paper PDF and slides retain their original CEUR-WS CC-BY 4.0 licensing where applicable.
