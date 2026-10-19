# Using NLIDNet

Run commands from the repository root. The package provides a unified `nlidnet` interface; `python -m nlidnet` is equivalent.

## Installation and data preparation

Python 3.10 or later is required. Install PyTorch and torchvision for your CUDA environment, then install this repository:

```bash
pip install -e .
```

Event/video baselines and RGB controls use the optional dependencies:

```bash
pip install -e ".[baselines]"
```

The [Hugging Face archive](https://huggingface.co/datasets/GWDx/NLID/tree/main) contains original recordings, not preprocessed training tensors or stage clips. Retain its liquid-folder layout under `data/raw/`:

```text
data/raw/water/event1.raw
data/raw/water/rgb1.avi
data/raw/ethanol_20/event1.raw
data/raw/ethanol_20/rgb1.avi
```

`eventN.raw` is the event stream; `rgbN.avi` is the corresponding RGB recording from that trial. The event model uses the RAW stream. RGB baselines and appearance controls read AVI videos directly. Decoding Prophesee `.raw` files requires the Metavision SDK; NPY, NPZ and safetensors event arrays can be read without the camera SDK.

First index independent trials, using recorded split assignments (`raw_path,split`, with optional `trial_id,prefix`). The archive's `metadata.csv` lists `file_name,category,sensor` and is not a training/test split file. Then decode the event streams and produce the model inputs:

```bash
python -m nlidnet data index-raw --raw-root data/raw --split-file data/splits.csv --output data/trials.csv
python -m nlidnet data prepare --raw-root data/raw --manifest data/trials.csv --output data/processed
```

If preparing a new dataset, `index-raw --test-per-class 10` creates a new split from naturally sorted filenames. It does not recover the manuscript's original split. `--folder-map` accepts a JSON mapping from canonical liquid names to source folder names.

When `rgbN.avi` is present beside `eventN.raw`, `index-raw` adds its relative `video_path` to the same trial manifest. Event and RGB workflows therefore share trial identifiers and split assignments. The label map covers the 30 liquid conditions used in this work; the archive's additional `medical_alcohol_75v` category is outside this protocol.

Each decoded event trial produces:

```text
data/processed/train/<prefix>.pt             Six ROI signals, [6,9000]
data/processed/train/<prefix>.safetensors    Integer x,y,t,p events
data/processed/train/<prefix>.txt            Raw liquid label
```

Test trials use `test/`. The trial schema is shown in [data/trials.example.csv](../data/trials.example.csv). Existing processed indices preserve filenames; their generated trial IDs are `split_prefix`. Use the recorded physical trial IDs in an existing manifest when available.

The `.safetensors` file retains the original integer `x,y,t,p` events. The `.pt` file stores six ROI count signals in 9000 bins of 2 ms, starting at the recording's first event. The training loader slices a 700-bin stage window at bin 2950, 3650 or 4350, with the configured temporal offset. All windows and offsets from one physical trial remain in that trial's split. Event-baseline representations use the same stage boundaries when exported with `data represent`.

For an existing processed dataset, indexing is available without decoding RAW again:

```bash
python -m nlidnet data index-processed --event-root data/processed --output data/trials.csv
```

Choose either `index-raw` or `index-processed` for the output manifest; commands refuse to overwrite an existing file. `index-processed` reconstructs event-tensor pairings only. Preserve the original raw manifest when using its RGB paths or recorded trial IDs.

```bash
python -m nlidnet data validate
python -m nlidnet data preview --events data/processed/test/water_event1.safetensors
python -m nlidnet data represent --manifest data/trials.csv --event-root data/processed --representation time_surface
```

`represent` also supports `event_frame` and `voxel_grid`, event-count or temporal partitioning, custom ROIs, partial event segments and shifted views. `prepare --labels 25 26 27 28 29` processes added ethanol concentrations without changing the label map.

## Training, adaptation and comparison experiments

Run commands from the repository root. [configs/paper.yaml](../configs/paper.yaml) contains the manuscript settings: 150 source-training epochs, supervised contrastive weight 0.02, physical supervision weight 0.10 and attention variance weight 0.0005. Stage windows start at bins 2950, 3650 and 4350 and contain 700 bins of 2 ms each.

Place `resnet34_imagenet1k_v1.pth` and `evflownet.pth` in `checkpoints/`, or set their paths in the configuration. Evaluation checkpoints include the optical-flow network.

```bash
python -m nlidnet train --stage stage1
python -m nlidnet evaluate --stage stage1 --checkpoint outputs/stage1/pretrain/final.pth
python -m nlidnet adapt classification --stage stage1 --checkpoint outputs/stage1/pretrain/final.pth --method linear_probe --shots 1 5 10 20 --seeds 1 2 3
python -m nlidnet adapt concentration --stage stage1 --checkpoint outputs/stage1/pretrain/final.pth --method full_ft --shots 5 10 --seeds 1
```

Adaptation supports `linear_probe`, `full_ft` and `no_task_pretrain`. Support sets are selected by trial and nested across shot counts for each seed. `--support-order` accepts the recorded support-order CSV. Classification probes use 500 epochs; regression probes and full fine-tuning use 100 epochs.

Cached features can be used to compare linear and MLP heads:

```bash
python -m nlidnet adapt classification --cached --train-features train_views.npz --test-features test_views.npz --heads linear mlp --shots 1 5 10 20 --seeds 1 2 3
```

Event baselines include AlexNet-LSTM, MobileNetV3-Chem and SpikingResformer; video baselines include VideoMAE, Go With the Flow and Don't Look Twice. Their manifests use `representation_path` or `video_path`, respectively; see [data/baselines.example.csv](../data/baselines.example.csv).

```bash
python -m nlidnet baselines train --model alexnet_lstm --stage stage1 --manifest data/representations/stage1/trials.csv --data-root data/representations/stage1
python -m nlidnet baselines evaluate --model alexnet_lstm --stage stage1 --manifest data/representations/stage1/trials.csv --data-root data/representations/stage1 --checkpoint outputs/baselines/alexnet_lstm/stage1/final.pth
python -m nlidnet ablations architecture --ablation no_flow --stage stage1
python -m nlidnet ablations architecture --ablation uniform --sample-count 50 --stage stage1
python -m nlidnet ablations architecture --ablation sparse --sample-count 50 --sparse-bin-us 200 --stage stage1 --accumulation-steps 4
python -m nlidnet ablations loss-weights --physics 0.1 0.3 --contrastive 0.02 --variance 0.0005 --stage stage1
```

Spatial-compression baselines from NLID include ConvTran, Transformer and InceptionTime. RGB appearance controls select the first frame, last frame or a frame during pouring, with optional grayscale conversion or a frozen backbone:

```bash
python -m nlidnet baselines control --kind temporal --model inception_time --stage stage1
python -m nlidnet baselines control --kind temporal --model convtran --task regression --stage stage2
python -m nlidnet baselines control --kind rgb --manifest data/trials.csv --video-root data/raw --frame-mode first_frame --initialization checkpoints/resnet34_imagenet1k_v1.pth --freeze-backbone
```

Video baselines also use the raw trial manifest and AVI paths, for example `python -m nlidnet baselines train --model videomae_v2 --stage stage1 --manifest data/trials.csv --data-root data/raw`. Stage frames are sampled by the video loader; there is no need to export separate AVI clips.

## Visualization and representation analysis

```bash
python -m nlidnet visualize attention --stage stage1 --checkpoint outputs/stage1/pretrain/final.pth --trial-id water_trial_002 --output outputs/attention
python -m nlidnet visualize gradcam --stage stage1 --checkpoint outputs/stage1/pretrain/final.pth --trial-ids water_trial_002 --output outputs/gradcam
python -m nlidnet visualize features --stage stage1 --checkpoint outputs/stage1/pretrain/final.pth --split train --output outputs/train_features
python -m nlidnet visualize export-attention --stage stage1 --checkpoint outputs/stage1/pretrain/final.pth --output outputs/temporal_attention
```

Feature export saves per-view embeddings in `views.npz` and trial-averaged embeddings in `embeddings.npz`. Grad-CAM export retains activations, gradients, raw maps and event backgrounds. `nlidnet visualize render-gradcam` renders these caches without rerunning the model. Temporal exports include encoder logits, smoothed logits, local attention, segment weights and selected bins. Composite figures can be rendered from the same caches:

```bash
python -m nlidnet visualize gallery --kind temporal --caches path/to/stage1/attention.npz path/to/stage2/attention.npz path/to/stage3/attention.npz --output outputs/temporal_gallery.png
python -m nlidnet visualize gallery --kind gradcam --caches path/to/water/gradcam.npz path/to/DMSO/gradcam.npz --output outputs/spatial_gallery.png
python -m nlidnet visualize gallery --kind embeddings --caches outputs/test_features/embeddings.npz --output outputs/liquid_embeddings.png
```

```bash
python -m nlidnet analyze physics --event-features event_features.pt --rgb-features rgb_features.pt
python -m nlidnet analyze embeddings --features outputs/test_features/embeddings.npz --reference-features outputs/train_features/embeddings.npz
python -m nlidnet analyze statistics --stage stage1 --shots 5
python -m nlidnet analyze speed --stage stage1 --checkpoint outputs/stage1/pretrain/final.pth
python -m nlidnet analyze flow --events data/processed/test/00000001.safetensors --stage stage1
python -m nlidnet analyze plot-embeddings --features outputs/test_features/embeddings.npz --method tsne
```

Videos can be assembled from existing frame sequences or paired event/flow caches. Exported frames contain no titles, timestamps or text overlays; timing and display settings are saved in the accompanying JSON file:

```bash
python -m nlidnet visualize video --frames path/to/event_frames --output outputs/event_pouring.mp4 --fps 50
python -m nlidnet analyze flow --events data/processed/test/00000001.safetensors --dense --all-stages --context-px 64 --output outputs/flow_cache
python -m nlidnet visualize flow-video --cache-root outputs/flow_cache --output outputs/event_flow.mp4 --gif outputs/event_flow.gif --slowdown 5
```

MP4 export uses H.264 with browser-compatible pixel format. Install FFmpeg or `pip install -e ".[media]"`. Event-frame timing uses the frame index and bin width; flow-cache timing comes from each stage's `meta.json`.

`physics` performs PCA-CCA with permutation and bootstrap analysis. `statistics` tests low-order event descriptors. `embeddings` reports class separation and reference-class distances. Each command lists its options through `--help`.
