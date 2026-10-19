# NLIDNet

Official implementation of **Active Hydrodynamic Interrogation Enables Embodied Liquid Sensing**.

Yuliang Wu†, Wei Zhai†, Wenrui Chang†, Wendi Gan, Chenfeng Wang, Yang Cao, Zheng-Jun Zha

University of Science and Technology of China. † Equal contribution.

[Dataset](https://huggingface.co/datasets/GWDx/NLID) · [Hardware](hardware/README.md)

NLIDNet encodes event streams recorded during robotic pouring into 256-dimensional liquid representations. Temporal attention selects informative moments from six ROI signals, and two branches encode flow patterns and velocity. We evaluate the representation on liquid classification, few-shot adaptation and ethanol concentration regression.

![NLIDNet](assets/architecture.png)

## Experiment entry points

| Experiment | Directory | Entry points |
| --- | --- | --- |
| Liquid classification | [classification](experiments/classification/) | `train.py`, `evaluate.py` |
| Few-shot classification | [few_shot](experiments/few_shot/) | `train.py`, `probe.py` |
| Concentration regression | [concentration](experiments/concentration/) | `train.py`, `probe.py` |
| Architecture and loss ablations | [ablations](experiments/ablations/) | `run.py`, `loss_weights.py` |
| Event and RGB baselines | [baselines](experiments/baselines/) | `prepare.py`, `train.py`, `evaluate.py` |
| Representation and sensing analysis | [analysis](experiments/analysis/) | See the analysis table below |
| Data preparation | [data](experiments/data/) | `prepare.py`, `validate.py` |

```text
experiments/        Experiment entry points, grouped by task
nlidnet/           Shared implementation
  models/          NLIDNet, temporal encoding, optical flow and baseline networks
  training/        Source training, adaptation, cached probes and losses
  data/            Event loading, preprocessing and sampling
  analysis/        Feature diagnostics, CCA, event statistics and timing
  visualization/   Attention, Grad-CAM and embedding plots
configs/           Experiment configuration
data/              Dataset manifests and reference annotations
hardware/          Camera acquisition and robot program
```

The main network is [nlidnet/models/network.py](nlidnet/models/network.py). Its branches are named `temporal_encoder`, `pattern_encoder` and `optical_flow`. Earlier checkpoint keys are accepted by the checkpoint loader.

## Installation

Python 3.10 or later is required. Install PyTorch and torchvision for your CUDA environment, then run:

```bash
pip install -e .
```

For the baseline models:

```bash
pip install -e ".[baselines]"
```

## Data

Download the data from [GWDx/NLID](https://huggingface.co/datasets/GWDx/NLID). Each row in `data/trials.csv` identifies one complete pouring trial:

```csv
trial_id,split,label,prefix,raw_path
water_001,train,0,00000000,water/event1.raw
water_002,test,0,00000001,water/event2.raw
```

`trial_id` is unique across splits; `prefix` is the tensor filename stem. All temporal views of a trial belong to the same split. Example manifests are provided in [data/](data/).

To process RAW recordings, install the Metavision SDK and run:

```bash
python -m experiments.data.prepare --manifest data/trials.csv --raw-root /path/to/NLID_RAW --output data/processed
```

Each trial produces an event tensor (`.safetensors`, `[N,4]` with columns `x,y,t,p`), six ROI count signals (`.pt`, `[6,9000]`) and a label file (`.txt`). Files are stored under `train/` or `test/`. Timestamps are in microseconds, polarity is 0 or 1, and ROI signals use 2 ms bins from the first event.

Set `event_root`, `roi_root` and `manifest` in [configs/paper.yaml](configs/paper.yaml). Event and ROI tensors may be stored in separate roots, with matching prefixes. Label order and ROI coordinates are defined in [protocol.py](nlidnet/protocol.py).

| Task | Labels |
| --- | --- |
| Base classification | 0–11, 21–24 |
| Few-shot classification | 12–20 |
| Ethanol regression | 0–5, 25–29 |

## Training and evaluation

Place the initialization weights in `checkpoints/`, or set their paths in the configuration:

| File | Initialization |
| --- | --- |
| `resnet34_imagenet1k_v1.pth` | ImageNet ResNet34 |
| `evflownet.pth` | Frozen optical-flow network |

```bash
python -m experiments.data.validate
python -m experiments.classification.train --stage stage1
python -m experiments.classification.evaluate --stage stage1 --checkpoint outputs/stage1/pretrain/final.pth
```

Use `stage2` or `stage3` for the other observation windows. Their start bins are 2950, 3650 and 4350, respectively; each window contains 700 bins. Training uses 150 epochs, AdamW at 1e-4, weight decay 0.05 and batches of five classes with two trials per class. Loss weights are 0.02 for supervised contrastive learning, 0.10 for physics supervision and 0.0005 for attention variance.

Evaluation averages logits over offsets `0,5,...,45` within each trial. The final epoch is used for evaluation. Complete model checkpoints contain the flow network and do not require separate initialization weights.

Use `--device cuda:1` to select a GPU, or `--device cpu` for CPU execution. Commands are run from the repository root.

## Few-shot adaptation

```bash
python -m experiments.few_shot.train --stage stage1 --checkpoint outputs/stage1/pretrain/final.pth --method linear_probe --shots 5 --seeds 1
python -m experiments.concentration.train --stage stage1 --checkpoint outputs/stage1/pretrain/final.pth --method full_ft --shots 10 --seeds 1
```

Support sizes are 1, 5, 10 and 20 trials per label. Support sets are nested across shot settings and shared across adaptation methods for the same seed. `--support-order` accepts a CSV with `rank` and `prefix` columns.

| Method | Initialization | Epochs |
| --- | --- | --- |
| `linear_probe` | Frozen pretrained encoder and a new linear head | 500 for classification; 100 for regression |
| `full_ft` | Pretrained encoder and a new head | 100 |
| `no_task_pretrain` | ImageNet ResNet34 and pretrained FlowNet | 100 |

Omit `--checkpoint` for `no_task_pretrain`. FlowNet remains frozen in all settings. Regression targets are concentration divided by 100; MSE is reported in squared percentage points.

Both training entries accept multiple `--shots` and `--seeds`. To compare linear and MLP heads using cached embeddings:

```bash
python -m experiments.few_shot.probe --train-features train_features.npz --test-features test_features.npz --heads linear mlp --shots 1 5 10 20 --seeds 1 2 3
python -m experiments.concentration.probe --train-features train_features.npz --test-features test_features.npz --heads linear mlp --shots 5 10 --seeds 1
```

Probe options include learning rate, weight decay, dropout, label smoothing and feature normalization. A support shot is counted by trial ID, including when a cache contains multiple views per trial.

## Baselines and ablations

```bash
python -m experiments.baselines.prepare --manifest data/trials.csv --event-root data/processed --stage stage1 --output data/baselines
python -m experiments.baselines.train --model alexnet_lstm --stage stage1 --manifest data/baselines/stage1/trials.csv --data-root data/baselines/stage1
python -m experiments.baselines.evaluate --model alexnet_lstm --stage stage1 --manifest data/baselines/stage1/trials.csv --data-root data/baselines/stage1 --checkpoint outputs/baselines/alexnet_lstm/stage1/final.pth
```

Available models are `alexnet_lstm`, `mobilenetv3_chem`, `spiking_resformer`, `videomae_v2`, `go_with_the_flow` and `dont_look_twice`. Event baselines use a `representation_path` manifest column; RGB baselines use `video_path`. VideoMAE-based models use `videomae_v2_base_kinetics.pth` for initialization.

```bash
python -m experiments.ablations.run --ablation no_flow --stage stage1
python -m experiments.ablations.run --ablation uniform --sample-count 50 --stage stage1
```

Uniform temporal sampling supports 20, 50 or 100 bins. Sparse sampling uses 20 or 200 microsecond sub-bins:

```bash
python -m experiments.ablations.run --ablation sparse --sample-count 50 --sparse-bin-us 200 --stage stage1 --accumulation-steps 4
python -m experiments.ablations.loss_weights --physics 0.1 0.3 --contrastive 0.02 --variance 0.0005 --stage stage1
```

`--train-offset-count`, `--learning-rate` and `--epochs` configure temporal augmentation and optimization. Add `--evaluate --checkpoint <path>` to evaluate an architecture ablation. The loss-weight entry supports `--dry-run` to inspect the run configurations.

## Visualization

```bash
python -m experiments.analysis.export_features --stage stage1 --checkpoint outputs/stage1/pretrain/final.pth --output outputs/embeddings
python -m experiments.analysis.attention --stage stage1 --checkpoint outputs/stage1/pretrain/final.pth --trial-id water_002 --output outputs/attention
python -m experiments.analysis.gradcam --stage stage1 --checkpoint outputs/stage1/pretrain/final.pth --trial-id water_002 --output outputs/gradcam
```

Feature export includes per-trial embeddings and a PCA plot. Attention and Grad-CAM commands use a trial ID from the test manifest.

## Additional analyses

These entries operate on recordings, checkpoints or exported features. Outputs are written under `outputs/`.

| Analysis | Entry point | Inputs |
| --- | --- | --- |
| Physical-property correlation (PCA–CCA, permutation and bootstrap) | [physics_correlation.py](experiments/analysis/physics_correlation.py) | Event and RGB feature files |
| Silhouette, class centroids and nearest-class distances | [embedding_quality.py](experiments/analysis/embedding_quality.py) | Features; optional reference features |
| Inference latency and GPU memory | [inference_speed.py](experiments/analysis/inference_speed.py) | Checkpoint and test data |
| Low-order event-statistics classifier | [event_statistics.py](experiments/analysis/event_statistics.py) | Event data; optional few-shot support size |
| Optical-flow extraction | [optical_flow.py](experiments/analysis/optical_flow.py) | Event tensor and flow checkpoint |
| PCA and t-SNE | [plot_embeddings.py](experiments/analysis/plot_embeddings.py) | Exported embeddings |
| Batch temporal-attention export | [export_attention.py](experiments/analysis/export_attention.py) | Checkpoint and event data |

```bash
python -m experiments.analysis.physics_correlation --event-features event_features.pt --rgb-features rgb_features.pt --output outputs/cca
python -m experiments.analysis.embedding_quality --features test_features.npz --reference-features train_features.npz
python -m experiments.analysis.inference_speed --checkpoint outputs/stage1/pretrain/final.pth --stage stage1
python -m experiments.analysis.event_statistics --stage stage1 --shots 5
python -m experiments.analysis.optical_flow --events data/processed/test/00000001.safetensors --stage stage1
python -m experiments.analysis.plot_embeddings --features outputs/embeddings/embeddings.npz --method tsne
python -m experiments.analysis.export_attention --checkpoint outputs/stage1/pretrain/final.pth --stage stage1
```

Feature analysis accepts NPZ, PT and CSV exports with features, raw labels and trial identifiers. Physical correlation uses class-mean features for the 21 liquid conditions with reference properties. Timing measures synchronized model forward passes. Each entry lists its arguments through `--help`.

## Acquisition platform

![Acquisition platform](assets/acquisition_platform.png)

Camera configuration, acquisition commands and the UR5e program are provided in [hardware/](hardware/README.md).

## Citation

```bibtex
@misc{wu_nlidnet,
  title  = {Active Hydrodynamic Interrogation Enables Embodied Liquid Sensing},
  author = {Wu, Yuliang and Zhai, Wei and Chang, Wenrui and Gan, Wendi and Wang, Chenfeng and Cao, Yang and Zha, Zheng-Jun},
  url    = {https://github.com/Changwenrui/NLIDNet},
  note   = {Manuscript}
}
```

## License and acknowledgements

This project is released under the [MIT License](LICENSE). We thank the authors of ConvTran, EVFlowNet, VideoMAE and SpikingResformer. Acquisition software and platform figures are also available in our [NLID repository](https://github.com/wyl-OvO/NLID). See [NOTICE](NOTICE) for attribution.
