# NLIDNet

### Active Hydrodynamic Interrogation Enables Embodied Liquid Sensing

Yuliang Wu†, Wei Zhai†, Wenrui Chang†, Wendi Gan, Chenfeng Wang, Yang Cao, Zheng-Jun Zha

University of Science and Technology of China &nbsp; | &nbsp; † Equal contribution

**[Hugging Face Dataset](https://huggingface.co/datasets/GWDx/NLID)** &nbsp; | &nbsp; [Video](#video) &nbsp; | &nbsp; [Method](#method) &nbsp; | &nbsp; [Code](#code) &nbsp; | &nbsp; [Hardware](hardware/README.md) &nbsp; | &nbsp; [Citation](#citation)

**A robot can learn about a liquid by eliciting and observing its motion.** A repeatable pouring action reveals transient differences in jet formation, impact, entrainment and breakup that are difficult to see in static appearance. Event-based vision captures these dynamics, and NLIDNet learns a shared representation for liquid recognition, few-shot adaptation and concentration estimation.

![Active hydrodynamic interrogation: robotic pouring, event-based observation and liquid sensing](assets/overview.png)

## Video

**Event response during pouring.** Opposite event polarities are shown in red and blue. Playback follows the original sampling of the exported event snapshots.

https://github.com/user-attachments/assets/35b29a13-9c93-4753-ab63-834f00020acd

[Download MP4](assets/demos/event_pouring.mp4)

## Method

NLIDNet follows the liquid response through three complementary mechanisms:

- **Temporal selection.** Six regional event-count signals guide a ConvTran encoder. Attention is normalized within short temporal segments to select informative intervals.
- **Pattern and velocity encoding.** Attentive polarity maps describe the evolving interfaces; a frozen EVFlowNet and trainable velocity encoder describe their motion. The two branches form a 256-dimensional response embedding.
- **Representation learning and transfer.** Category supervision, supervised contrastive learning, attention variance regularization and auxiliary physical-attribute supervision shape the embedding. Linear probes or full fine-tuning adapt it to downstream tasks.

![NLIDNet: temporal selection, attentive event construction and two-stream response encoding](assets/architecture.png)

The mechanisms are implemented in [time_attention.py](nlidnet/models/time_attention.py), [event_representation.py](nlidnet/models/event_representation.py), [encoders.py](nlidnet/models/encoders.py) and [losses.py](nlidnet/training/losses.py). The complete model is assembled in [network.py](nlidnet/models/network.py).

## Dataset

**[Access the NLID dataset on Hugging Face →](https://huggingface.co/datasets/GWDx/NLID)**

This work builds on our [NLID project](https://github.com/wyl-OvO/NLID), sharing its robotic pouring platform and event/RGB recordings. The experiments study the following tasks:

| Task | Protocol |
| --- | --- |
| Liquid recognition | 16 base liquid categories |
| Few-shot recognition | Nine categories excluded from source-task training |
| Concentration estimation | Water and ten ethanol concentrations, 0-100% in 10% increments |

Each trial contains three 1.4 s observation windows, beginning at 5.90, 7.30 and 8.70 s, with 2 ms event bins. Predictions and few-shot support counts are defined by independent physical trials. Temporal views of a recording retain the same trial identifier.

<details>
<summary>Event and RGB observations of four liquids</summary>

Recorded pouring sequences from NLID: water, 30% glycerol, ethanol and DMSO. Event and RGB rows show the evolving liquid response at successive times.

![Event and RGB pouring observations for water, glycerol, ethanol and DMSO](assets/event_rgb_comparison.png)

</details>

Label mappings and manifest examples are provided in [data/](data/). The [data tools](nlidnet/data/) cover RAW decoding, six-ROI signals, trial indexing, polarity frames, time surfaces, voxel grids and validation. See the [data preparation workflow](docs/usage.md#installation-and-data-preparation) for conversion and indexing commands.

## Acquisition platform

A UR5e robot executes a prescribed pouring trajectory. A beam splitter aligns a Prophesee EVK4 event camera and an iRayple RGB camera to observe the same scene.

![Robotic pouring platform with aligned event and RGB observations](assets/acquisition_platform.png)

Camera interfaces, acquisition scripts, platform dimensions and the robot program are available in [hardware](hardware/README.md).

## Code

The `nlidnet` command connects data preparation, training, adaptation and analysis. Implementation modules are organized by function:

```text
NLIDNet/
  nlidnet/
    commands/        Command-line workflows
    models/          Time attention, event construction and two-stream encoders
    data/            RAW decoding, trial indexing and representations
    training/        Objectives, source training and adaptation
    analysis/        Correlation, statistics, embeddings and timing
    visualization/   Attention, Grad-CAM, galleries and video export
  configs/           Paper configuration
  docs/              Installation and workflow reference
  data/              Manifest examples, labels and reference annotations
  hardware/          Camera interfaces and robot program
  assets/            Article figures and demonstration videos
  tests/             Functional checks
```

| Workflow | Command | Implementation |
| --- | --- | --- |
| RAW processing and indexing | `nlidnet data` | [data commands](nlidnet/commands/prepare_data.py) |
| Source training | `nlidnet train` | [training](nlidnet/training/train.py) |
| Trial-level evaluation | `nlidnet evaluate` | [evaluation](nlidnet/training/evaluate.py) |
| Few-shot classification and concentration regression | `nlidnet adapt` | [adaptation](nlidnet/training/adapt.py), [cached probes](nlidnet/training/probes.py) |
| Event, video, temporal and RGB controls | `nlidnet baselines` | [baseline models](nlidnet/models/baselines/) |
| Architecture and loss ablations | `nlidnet ablations` | [ablation models](nlidnet/models/ablations.py) |
| Representation analysis | `nlidnet analyze` | [analysis](nlidnet/analysis/) |
| Figures and video export | `nlidnet visualize` | [visualization](nlidnet/visualization/) |

`python -m nlidnet` provides the same interface. Run `nlidnet --help` or add `--help` after a command to inspect its options.

[Installation, data preparation and experiment commands](docs/usage.md)

## Citation

```bibtex
@misc{wu_nlidnet,
  title  = {Active Hydrodynamic Interrogation Enables Embodied Liquid Sensing},
  author = {Wu, Yuliang and Zhai, Wei and Chang, Wenrui and Gan, Wendi and Wang, Chenfeng and Cao, Yang and Zha, Zheng-Jun},
  url    = {https://github.com/Changwenrui/NLIDNet},
  note   = {Manuscript}
}
```

## Acknowledgements

This work extends our [NLID project](https://github.com/wyl-OvO/NLID), sharing the pouring platform, recordings and event/RGB visualizations. We thank the authors of ConvTran, EVFlowNet, InceptionTime, VideoMAE and SpikingResformer. Source attribution is provided in [NOTICE](NOTICE).

Released under the [MIT License](LICENSE).
