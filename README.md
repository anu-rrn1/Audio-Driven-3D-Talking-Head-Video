# TinyPointTalk

Audio-driven 3D talking heads at laptop scale, inspired by PointTalk [1]. Given a 3D face mesh of
a person, a few photos of them and a speech recording, the system produces a video of a
relightable 3D Gaussian head that speaks the audio.

The system is a FLAME-based pipeline.

## Method overview

```
 identity mesh ──► 02 registration to FLAME ──► rigged identity + mouth region + cavity/teeth
                                                        │
                    03 lip point cloud  P = B·V  ◄───────┤   (points ⇄ mesh by Laplacian editing)
                                                        │
 VOCASET ──► 04a PCA motion basis ──► 04b Audio2Motion ──► 04c articulation gain
                                                        │
 1-5 photos ──► 05 landmark fit (pose, expression, jaw) + albedo / SH lighting
                                                        │
                    06 mesh-rigged Gaussian avatar ◄────┘
                                                        │
 speech audio ──────────────────────────────► 07 audio → 3D motion → rendered video
```

## Repository layout

```
tinypt/                    library
  geometry/                mesh.py, point_cloud.py, registration.py, mouth.py
  flame/                   model.py (FLAME in PyTorch), landmarks.py (MediaPipe embedding)
  audio/                   features.py, audio2motion.py (network), vocaset.py (motion dataset)
  rendering/               rasterizer.py, splatting.py, gaussian_avatar.py
  visualization.py, subtitles.py, metrics.py, io_utils.py, config.py
scripts/flame_pipeline/    steps 01-07 of the pipeline
configs/default.yaml       all hyper-parameters
notebooks/                 GPU training of step 4b (Colab: v1, Kaggle: v2; steps 04c and 07 use v1)
tests/test_geometry.py     unit checks of the geometry and FLAME code
outputs/                   results of every step (figures, metrics, videos)
data/                      inputs and intermediate files
```

Each script reads its inputs from `data/` (or an earlier step's outputs), writes
intermediate arrays to `data/identity/`, and writes figures, metrics
and videos to `outputs/flame_pipeline/<step>/`.

## Setup

Python 3.10 or 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate                # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python tests/test_geometry.py         # needs no external data
```

External data (free registration, not redistributable):

- FLAME 2020 from https://flame.is.tue.mpg.de: put `generic_model.pkl` in `data/flame/`.
  `FLAME_masks.pkl` is optional; without it the lip region is derived from the mouth-hole
  boundary of the template.
- VOCASET from https://voca.is.tue.mpg.de, for steps 4a-4c and 7: put it in `data/vocaset/`.

## Running

```bash
python run_pipeline.py                         # all steps of the FLAME pipeline
python run_pipeline.py --from 05               # resume from step 05
```

Single steps run as scripts, e.g. `python scripts/flame_pipeline/07_generate_video.py --wav speech.wav`.
To use real inputs, pass your own mesh to step 02 (`--mesh face.obj`) and a photo folder to
step 05 (`--photos <folder>`). Step 4b is best trained on a GPU with one of the notebooks.

## Results

All results come from the synthetic identity of step 01 and the unseen VOCASET test speakers.
Figures and full metrics are in `outputs/`.

## Limitations

- Evaluation of the photo and avatar steps uses a synthetic identity; real scans and photos
  go through the same code but have not been evaluated quantitatively.
- The audio model is trained on 8 VOCASET speakers and under-articulates on unseen speakers.
- Rendering runs on the CPU at 256 × 256 pixels (about 1 s per frame).

## References

[1] Y. Xie, T. Feng, X. Zhang, X. Luo, Z. Guo, W. Yu, H. Chang, F. Ma and F. R. Yu.
    *PointTalk: Audio-Driven Dynamic Lip Point Cloud for 3D Gaussian-based Talking Head
    Synthesis.* Proceedings of the AAAI Conference on Artificial Intelligence, 39(8),
    8753-8761, 2025. [arXiv:2412.08504](https://arxiv.org/abs/2412.08504)