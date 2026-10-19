# NLIDNet

## Active Hydrodynamic Interrogation Enables Embodied Liquid Sensing

Yuliang Wu†, Wei Zhai†, Wenrui Chang†, Wendi Gan, Chenfeng Wang, Yang Cao, Zheng-Jun Zha

University of Science and Technology of China | † Equal contribution

[Videos](#videos) | [Method](#method) | [Visualizations](#reading-the-learned-response) | [Dataset](https://huggingface.co/datasets/GWDx/NLID) | [Hardware](hardware/README.md) | [Code](#code)

Transparent liquids may look nearly identical at rest, yet respond differently when accelerated, deformed and poured. We study whether a robot can create an informative sensing signal through its own manipulation action. A repeatable pouring trajectory elicits a transient hydrodynamic response; event-based vision records its evolution; NLIDNet encodes the response into a compact liquid representation.

The observation is the liquid's response to a prescribed action: jet formation, impact, entrainment, breakup and surface relaxation. Temporal selection and two complementary visual branches allow the encoder to follow both the morphology and motion of these evolving interfaces. The resulting representation is used for liquid classification, adaptation to previously unseen categories and ethanol concentration regression.

![Active hydrodynamic interrogation and the three stages of pouring](assets/overview.png)

- **Action as measurement.** A standardized robotic pour supplies the stimulus that reveals otherwise weak appearance differences.
- **Event-based response readout.** Asynchronous brightness changes capture rapidly evolving jets, droplets and liquid interfaces.
- **Selective temporal encoding.** Six regional signals guide event aggregation and the choice of motion-estimation intervals.
- **A shared response representation.** Pattern and velocity descriptors form a 256-dimensional embedding for multiple downstream tasks.

## Videos

### Event response during pouring

A recorded pouring sequence viewed through the event camera. Red and blue indicate opposite event polarities. The video includes the recording timestamp and uses the original temporal sampling of the exported snapshots.

https://github.com/user-attachments/assets/c12133d8-801f-491e-a1a2-0421f549f4e0

### Event counts and estimated optical flow

Paired event counts and frozen EVFlowNet estimates from the same recording, covering all three stages. Playback is five times slower than the recorded process. Flow hue indicates direction and brightness indicates magnitude, with one display scale shared across stages.

https://github.com/user-attachments/assets/3ac0531d-4cbe-4bfb-9ef6-c6cb29ca0284

MP4 copies are retained in [assets/demos](assets/demos/).

## The pouring response

Pouring produces several regimes within one trial. The initial liquid jet forms and strikes the receiving vessel; continued pouring develops entrainment and circulation; the later response includes column breakup, droplets and weak surface motion. These stages provide different views of the same elicited liquid response.

| Observation window | Recording time | Representative dynamics |
| --- | --- | --- |
| Stage 1 | 5.90-7.30 s | Jet formation, impact, splashing and wall spreading |
| Stage 2 | 7.30-8.70 s | Developed impingement, entrainment and mixing |
| Stage 3 | 8.70-10.10 s | Column breakup, droplet impact and surface relaxation |

Each nominal window contains 700 bins at 2 ms resolution. Temporal views within a trial shift the observation window while retaining the same physical trial identifier.

Event and RGB observations from our NLID acquisition platform illustrate the different appearances of the evolving flow:

![Event and RGB observations of different liquids during pouring](assets/event_rgb_comparison.png)

## Method

NLIDNet maps the event stream to a response embedding through temporal attention, attentive event representations and a two-stream encoder.

![NLIDNet temporal selection and two-stream response encoder](assets/architecture.png)

### Regional signals and temporal selection

The observation region is divided into six fixed ROIs. Counting events in each ROI produces a `6 x 700` signal for a stage window. A ConvTran-based time encoder predicts a score for every fine temporal bin. Gaussian smoothing connects nearby scores before attention is computed.

The window is divided into 20 segments, each containing 35 fine bins. Attention is normalized within each segment:

$$
\alpha_{m,j}=\operatorname{softmax}_{j}(s_{m,j}),
\qquad
j_m^{*}=\arg\max_j\alpha_{m,j}.
$$

Local weights guide event-map construction. Selected bins identify adjacent intervals for optical-flow estimation. A second set of weights, obtained from segment scores, controls velocity-feature aggregation. The implementation is in [time_attention.py](nlidnet/models/time_attention.py).

### Attentive pattern and velocity encoding

The **pattern branch** accumulates positive and negative events using the learned local weights, yielding 20 polarity feature maps. A convolutional downsampler and ResNet34 encode their spatial structure. Segment features are averaged to summarize flow morphology.

The **velocity branch** constructs four-channel inputs containing polarity counts and normalized event timestamps. Two adjacent inputs are selected for each segment. A frozen EVFlowNet estimates motion, a trainable convolutional encoder extracts velocity descriptors, and segment attention combines them over time.

Pattern and velocity descriptors are concatenated and projected into a 256-dimensional response embedding. Classification and physical-attribute heads operate on this shared representation. See [event_representation.py](nlidnet/models/event_representation.py), [encoders.py](nlidnet/models/encoders.py) and the assembled [network.py](nlidnet/models/network.py).

### Representation learning and adaptation

Source training combines category supervision, supervised contrastive learning, attention variance regularization and auxiliary physical-attribute supervision:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{cls}}
+\lambda_{\mathrm{con}}\mathcal{L}_{\mathrm{con}}
+\lambda_{\mathrm{var}}\mathcal{L}_{\mathrm{var}}
+\lambda_{\mathrm{phys}}\mathcal{L}_{\mathrm{phys}}.
$$

The auxiliary targets are standardized reference values of density, viscosity and surface tension for the base categories. Downstream heads adapt the embedding to novel-category recognition or concentration regression. The encoder can remain frozen for a linear probe or be updated through full fine-tuning. The training objective is implemented in [losses.py](nlidnet/training/losses.py).

