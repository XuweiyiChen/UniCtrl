# UniCtrl

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/abs/2403.02332)
[![Project Page](https://img.shields.io/badge/Project-Website-5B7493?logo=googlechrome&logoColor=5B7493)]([https://tianxingwu.github.io/pages/FreeInit/](https://unified-attention-control.github.io/))
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-%20Hugging%20Face-ED7D31)](https://huggingface.co/spaces/Xuweiyi/UniCtrl)


This repository is the implementation of

### UniCtrl: Improving the Spatiotemporal Consistency of Text-to-Video Diffusion Models via Training-Free Unified Attention Control

- **Authors**: [Xuweiyi Chen](https://xuweiyichen.github.io/)<sup>\*</sup>, Tian Xia<sup>\*</sup>, [Sihan Xu](https://sihanxu.github.io/)<sup>**</sup>
- **Affiliation**: PixAI.art, University of Michigan
- <sup>*</sup>*Equal contribution*, <sup>**</sup>*Correspondence*

### [Project page](https://unified-attention-control.github.io/) | [Paper](https://arxiv.org/abs/2403.02332) | Demo
<table>
  <tr>
    <td><img src="./assets/girl/orig_sample.gif" alt="Original" style="width:100%"></td>
    <td><img src="./assets/girl/ctrl_sample.gif" alt="UniCtrl" style="width:100%"></td>
  </tr>
  <tr>
    <td align="center">Original</td>
    <td align="center">UniCtrl</td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="./assets/mt/orig_sample.gif" alt="Original" style="width:100%"></td>
    <td><img src="./assets/mt/ctrl_sample.gif" alt="UniCtrl" style="width:100%"></td>
  </tr>
  <tr>
    <td align="center">Original</td>
    <td align="center">UniCtrl</td>
  </tr>
</table>

## Updates🔥 

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
