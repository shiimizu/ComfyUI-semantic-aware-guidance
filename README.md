# Semantic-aware Guidance  (S-CFG)

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) node for Semantic-aware Guidance based on the paper "Rethinking the Spatial Inconsistency in Classifier-Free Diffusion Guidance"

[Paper](https://arxiv.org/abs/2404.05384) | [Code](https://github.com/SmilesDZgk/S-CFG) | [sd-webui-incantations](https://github.com/v0xie/sd-webui-incantations) <sup>(SD-WebUI extension)</sup>

Dynamically rescale CFG guidance per semantic region to a uniform level to improve image / text alignment.


Other details:
* Computationally expensive. Large resolutions (like on an upscale pass) can trigger OOM. 
* made to work with other resolutions
* made to work with SDXL (based on guesswork, since the original paper/code didn't mention SDXL)
* only works for models with a U-Net backbone (i.e. SD1.x - SDXL)
* uses `sampler_cfg_function` (there can only be one node active using this, so it may conflict with other nodes using the same thing)

## Installation

You can either:

* In a termnial, run: `git clone https://github.com/shiimizu/ComfyUI-semantic-aware-guidance.git` into your `ComfyUI/custom_nodes/` folder.

* Install it via [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager) (search for custom node named "s-cfg").
