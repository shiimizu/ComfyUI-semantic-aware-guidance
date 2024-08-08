# Semantic-aware Guidance  (S-CFG)

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) node for Semantic-aware Guidance based on the paper "Rethinking the Spatial Inconsistency in Classifier-Free Diffusion Guidance"

Paper: https://arxiv.org/abs/2404.05384

Code: https://github.com/SmilesDZgk/S-CFG

SD-WebUI extension: [sd-webui-incantations](https://github.com/v0xie/sd-webui-incantations)

Computationally expensive. High resolutions (like on an upscale pass) can trigger OOM. Other details:
* made to work with other resolutions
* made to work with sdxl (based on guesswork, since the original paper/code didn't mention sdxl)
* uses `sampler_cfg_function` (there can only be one node that uses this in ComfyUI)
* only works for SD with a U-Net backbone (i.e. SD1.x - SDXL)

## Installation

* In a termnial, run: `git clone https://github.com/shiimizu/ComfyUI-semantic-aware-guidance.git` into your `ComfyUI/custom-nodes/` folder.
