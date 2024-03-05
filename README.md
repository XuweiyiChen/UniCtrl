# UniCtrl

This repository is the implementation of

**UniCtrl: Improving the Spatiotemporal Consistency of Text-to-Video Diffusion Models via Training-Free Unified Attention Control**  
Xuweiyi Chen*,  Tian Xia*,  Sihan XU**  
PixAI.art, University of Michigan  
(* Equal contribution,    ** Correspondence)

### [Project page](https://unified-attention-control.github.io/) | [Paper(https://arxiv.org/abs/2403.02332) | Demo

## 🔥 Updates

## 🔨 Quick Start

### 1. Clone Repo

```
git clone https://github.com/XuweiyiChen/UniCtrl.git
cd UniCtrl
cd examples/AnimateDiff
```

### 2. Prepare Environment

```
conda env create -f environment.yaml
conda activate animatediff_pt2
```

### 3. Download Checkpoints

Please refer to the [official repo](https://github.com/guoyww/AnimateDiff) of AnimateDiff. The setup guide is listed [here](https://github.com/guoyww/AnimateDiff/blob/main/__assets__/docs/animatediff.md).

### 🤗 Gradio Demo

We provide a Gradio Demo to demonstrate our method with UI.

```
python app.py
```
Alternatively, you can try the online demo hosted on Hugging Face: [[demo link]](https://huggingface.co/).

## :white_heart: Acknowledgement

This project is distributed under the MIT License. See `LICENSE` for more information.

The example code is built upon [AnimateDiff](https://github.com/guoyww/AnimateDiff) and [FreeInit](https://github.com/TianxingWu/FreeInit). Thanks to the team for their impressive work!
