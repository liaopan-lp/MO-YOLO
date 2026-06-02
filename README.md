# DecoderTracker: Decoder-Only Method for Multiple-Object Tracking

Official implementation of **"DecoderTracker: Decoder-Only Method for Multiple-Object Tracking"** published in *Pattern Recognition*.

📄 **Paper**: [DecoderTracker: Decoder-Only Method for Multiple-Object Tracking](https://www.sciencedirect.com/science/article/abs/pii/S0031320326002074)

📌 **arXiv**: [DecoderTracker: Decoder-Only Method for Multiple-Object Tracking](https://arxiv.org/abs/2310.17170)

---

## 📖 Introduction

DecoderTracker is a multi-object tracking framework that builds upon the YOLO/DETR architecture with a novel **Fixed-Size Query Memory (FSQM)** mechanism for efficient end-to-end tracking. Unlike traditional tracking-by-detection methods that rely on separate detection and association steps, DecoderTracker integrates detection and tracking into a unified decoder architecture.

### Key Features

- **Fixed-Size Query Memory (FSQM)**: A novel query memory management mechanism that maintains a fixed-size set of track and detection queries, enabling stable and efficient online tracking.
- **Three-Stage Training Pipeline**: A progressive training strategy from pure detection → weak tracking (TBSP) → full tracking (TALA), based on MOTR's training methodology.
- **Decoder-Only Architecture**: Tracking is performed entirely within the decoder, eliminating the need for explicit association post-processing (e.g., Hungarian matching, Re-ID features).
- **Comprehensive Evaluation Metrics**: Built-in support for HOTA, MOTA, MOTP, IDF1, and other standard MOT metrics following [TrackEval](https://github.com/JonathonLuiten/TrackEval).

---

## 🏗️ Architecture

The model follows a transformer-based encoder-decoder architecture:

1. **Backbone (YOLO-style)**: Extracts multi-scale features from input frames
2. **Encoder**: Processes features with deformable attention
3. **Decoder with FSQM**: Generates detection and tracking predictions using a fixed-size query memory
   - Track queries: Maintain identity across frames for tracked objects
   - Detection queries: Discover new objects not yet tracked
   - FSQM manages query lifecycle: initiation, update, and termination

---

## 🛠️ Installation

### Requirements

- **OS**: Linux or Windows
- **Python**: >= 3.8
- **CUDA**: >= 11.0
- **GPU**: NVIDIA GPU with >= 8GB VRAM (recommended: RTX 2080 Ti or better)

### Environment Setup

We recommend using the **August 2023** version of the Ultralytics environment. Create a conda environment:

```bash
conda create -n decoder-tracker python=3.10
conda activate decoder-tracker
```

Install PyTorch (adjust CUDA version as needed):

```bash
# For CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121
```

Install other dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- `torch >= 2.0`
- `torchvision >= 0.15`
- `opencv-python >= 4.6.0`
- `scipy >= 1.4.1`
- `matplotlib >= 3.2.2`
- `PyYAML >= 5.3.1`
- `tqdm >= 4.64.0`

---

## 📊 Dataset Preparation

### Supported Datasets

| Dataset | Config File | Description |
|---------|------------|-------------|
| MOT17 | `MOT.yaml` | Multiple Object Tracking benchmark |
| DanceTrack | `dancetracker.yaml` | Dance movement tracking benchmark |
| KITTI | `KITTI.yaml` | Autonomous driving tracking benchmark |

### Data Format

Organize datasets following the MOTR/FairMOT format with YOLO-compatible labels:

```
datasets/
├── MOT17/
│   ├── images/
│   │   ├── train/
│   │   │   ├── MOT17-02-DPM/
│   │   │   │   ├── img1/
│   │   │   │   │   ├── 000001.jpg
│   │   │   │   │   ├── 000002.jpg
│   │   │   │   │   └── ...
│   │   │   └── ...
│   │   └── val/
│   ├── labels_with_ids/
│   │   ├── train/
│   │   └── val/
│   └── train.txt
│   └── val.txt
├── dancetracker/
│   ├── train/
│   ├── val/
│   └── yolo_track/
│       ├── train.txt
│       └── val.txt
└── KITTI_TRACKING/
    ├── training/
    └── training_yolotrack_format/
        └── data/
            ├── train.txt
            └── val.txt
```

### Dataset Config

Edit the YAML config files to point to your dataset paths. For example, `dancetracker.yaml`:

```yaml
train: /path/to/dancetrack/yolo_track/train.txt
val: /path/to/dancetrack/yolo_track/val.txt

names:
  0: person
```

---

## 🚀 Training

DecoderTracker uses a **three-stage progressive training strategy**:

### Stage 1: Pure Detection Training
Train the network as a standard detector (no tracking). Every frame is treated as the first frame, allowing the network to learn object appearance features.

```python
from ultralytics import DecoderTracker

model = DecoderTracker("yolo_track.yaml", use_fsqm=True, training_stage=1)
model.train(data="dancetracker.yaml", epochs=50, batch=8)
```

### Stage 2: TBSP Training (Weak Tracking)
Introduce TBSP (Tracking Box Selection Process) as weak supervision. Track queries are maintained across frames, and detect queries overlapping with track queries are filtered out.

```python
model = DecoderTracker("yolo_track.yaml", use_fsqm=True, training_stage=2)
model.load("path/to/stage1_best.pt")
model.train(data="dancetracker.yaml", epochs=30, batch=1)
```

### Stage 3: TALA Training (Full Tracking)
Use MOTR's Tracklet-Aware Label Assignment (TALA) for comprehensive tracking performance. This stage fully trains the tracking capability.

```python
model = DecoderTracker("yolo_track.yaml", use_fsqm=True, training_stage=3)
model.load("path/to/stage2_best.pt")
model.train(data="dancetracker.yaml", epochs=30, batch=1)
```

### Full Pipeline Script

Use the provided `start_train.py` to run all three stages sequentially:

```bash
python start_train.py
```

Or customize individual stages in `start_train.py`:

```python
from start_train import train_stage_1, train_stage_2, train_stage_3, train_all_stages

# Run all stages
train_all_stages(data_config="dancetracker.yaml", device="0")

# Or run individual stages
s1 = train_stage_1(data_config="dancetracker.yaml", epochs=50, batch=8, device="0")
s2 = train_stage_2(data_config="dancetracker.yaml", stage1_weights=s1, epochs=30, device="0")
s3 = train_stage_3(data_config="dancetracker.yaml", stage2_weights=s2, epochs=30, device="0")
```

### Training Parameters

| Parameter | Description | Stage 1 | Stage 2 | Stage 3 |
|-----------|-------------|---------|---------|---------|
| `epochs` | Training epochs | 50 | 30 | 30 |
| `batch` | Batch size | 8 (can be larger) | 1 | 1 |
| `imgsz` | Input image size | 640 | 640 | 640 |
| `training_stage` | Training phase | 1 | 2 | 3 |
| `use_fsqm` | Enable FSQM | True | True | True |

> **Note**: Stage 2 and Stage 3 require `batch=1` because video-based training maintains tracking state across frames within a sequence.

---

## 🔍 Inference & Evaluation

### Inference

```python
from ultralytics import DecoderTracker

model = DecoderTracker("path/to/best.pt")
model("path/to/video_frames/", show=False, save=True)
```

### Evaluation with Tracking Metrics

```python
from ultralytics import DecoderTracker

model = DecoderTracker("path/to/best.pt")
metrics = model.val(data="MOT.yaml")
```

The evaluation outputs comprehensive tracking metrics per video and overall, including HOTA, MOTA, MOTP, IDF1, and other standard MOT metrics.

---

## 📁 Project Structure

```
DecoderTracker/
├── ultralytics/
│   ├── models/
│   │   ├── DecoderTracker/
│   │   │   ├── model.py          # DecoderTracker model class
│   │   │   ├── train.py          # Training logic
│   │   │   ├── val.py            # Validation with tracking metrics
│   │   │   └── predict.py        # Prediction/inference
│   │   └── ...
│   ├── nn/
│   │   ├── modules/
│   │   │   ├── head.py           # DecoderTracker head with FSQM
│   │   │   └── ...
│   │   └── tasks.py              # TrackingModel definition
│   └── utils/
│       ├── hota.py               # HOTA metric implementation
│       ├── clear.py              # CLEAR metrics (MOTA, MOTP, etc.)
│       ├── identity.py           # Identity metrics (IDF1, IDR, IDP)
│       └── ...
├── MOTR/                         # MOTR reference implementation
│   ├── models/                   # MOTR model components
│   ├── configs/                  # MOTR training configs
│   └── ...
├── start_train.py                # Three-stage training script
├── run_test.py                   # Testing/evaluation script
├── yolo_track.yaml               # Model architecture config
├── dancetracker.yaml             # DanceTrack dataset config
├── MOT.yaml                      # MOT17 dataset config
├── KITTI.yaml                    # KITTI dataset config
└── requirements.txt              # Python dependencies
```

---

## 🙏 Acknowledgments

This project builds upon the following excellent works:

- **[Ultralytics](https://github.com/ultralytics/ultralytics/)**: The YOLO framework providing the backbone and training infrastructure.
- **[MOTR](https://github.com/megvii-research/MOTR)**: End-to-End Multiple-Object Tracking with Transformer, providing the TALA training strategy and some modules.
- **[TrackEval](https://github.com/JonathonLuiten/TrackEval)**: Comprehensive tracking evaluation metrics.

---

## 📄 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{LIAO2026113242,
title = {DecoderTracker: Decoder-only end-to-end method for multiple-object tracking},
journal = {Pattern Recognition},
volume = {177},
pages = {113242},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2026.113242},
url = {https://www.sciencedirect.com/science/article/pii/S0031320326002074},
author = {Pan Liao and Feng Yang and Di Wu and Wenhui Zhao and Jinwen Yu and Dingwen Zhang},
}
```

```bibtex
@article{decodertracker2023,
  title={DecoderTracker: Decoder-Only Method for Multiple-Object Tracking},
  author={Pan Liao and others},
  journal={arXiv preprint arXiv:2310.17170},
  year={2023}
}
```

```bibtex
@inproceedings{zeng2021motr,
  title={MOTR: End-to-End Multiple-Object Tracking with TRansformer},
  author={Zeng, Fangao and Dong, Bin and Zhang, Yuang and Wang, Tiancai and Zhang, Xiangyu and Wei, Yichen},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

---

## 📜 License

This project is licensed under the [AGPL-3.0 License](LICENSE).
