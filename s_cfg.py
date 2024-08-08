"""
An unofficial implementation of "Rethinking the Spatial Inconsistency in Classifier-Free Diffusion Guidancee" for ComfyUI.

This builds upon the code provided in the official S-CFG repository: https://github.com/SmilesDZgk/S-CFG


@inproceedings{shen2024rethinking,
  title={Rethinking the Spatial Inconsistency in Classifier-Free Diffusion Guidancee},
  author={Shen, Dazhong and Song, Guanglu and Xue, Zeyue and Wang, Fu-Yun and Liu, Yu},
  booktitle={Proceedings of The IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year={2024}
}

Parts of the code are based on Diffusers under the Apache License 2.0:
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

import abc
import gc
import torch
import torch.nn.functional as F
from comfy.model_patcher import ModelPatcher
from operator import mul
from weakref import WeakSet
from comfy.ldm.modules.attention import optimized_attention_masked
from .utils import GaussianSmoothing, rep

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str, manual_r=None, count=True, r=None):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str, manual_r=None, count=True, r=None):
        if not count: return
        self.forward(attn, is_cross, place_in_unet, manual_r, count=count, r=r)
        # if self.cur_att_layer >= self.num_uncond_att_layers:
        #     self.forward(attn, is_cross, place_in_unet, manual_r, count=count, r=r)
        # self.cur_att_layer += 1
        # # reach the end
        # if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
        #     self.cur_att_layer = 0
        #     self.cur_step += 1
        #     self.between_steps()

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        # return {"down_cross": [], "mid_cross": [], "up_cross": [],
        #         "down_self": [], "mid_self": [], "up_self": []}
        return {"r2_cross": [],"r4_cross": [], "r8_cross": [], "r16_cross": [],
                "r2_self": [], "r4_self": [], "r8_self": [], "r16_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str, manual_r=None, count=True, r=None):  ####TODO 修改key名称
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if r is None:
            h = int(attn.size(1)**(0.5))
            self.H = max(self.H, h) # hack
            r = int(self.H/h)
        key = f"r{r}_{'cross' if is_cross else 'self'}"
        # print('=== attn.shape',attn.shape, key, self.H)
        # if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
        if r >= 2:  # avoid memory overhead
            if key not in self.step_store:
                self.step_store[key] = []
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        self.attention_store = self.step_store
        if self.save_global_store:
            with torch.no_grad():
                if len(self.global_store) == 0:
                    self.global_store = self.step_store
                else:
                    for key in self.global_store:
                        for i in range(len(self.global_store[key])):
                            self.global_store[key][i] += self.step_store[key][i].detach()
        for key, tensor_list in self.step_store.items():
            for tensor in tensor_list:
                del tensor
        del tensor_list
        torch.cuda.empty_cache()
        gc.collect()
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = self.attention_store
        return average_attention

    def get_average_global_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.global_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.H=0

    def __init__(self,H=0, save_global_store=False):
        '''
        Initialize an empty AttentionStore
        :param step_index: used to visualize only a specific step in the diffusion process
        '''
        super(AttentionStore, self).__init__()
        self.H = H
        self.save_global_store = save_global_store
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.global_store = {}
        self.curr_step_index = 0

class MyCrossAttnProcessor:
    def __init__(self, attention_store: AttentionStore, is_cross=False, manual_r=None):
        super().__init__()
        self.attention_store = attention_store
        self.is_cross = is_cross
        self.manual_r = manual_r

    def __call__(self, q, k, v, extra_options, mask=None):
        place_in_unet = extra_options["block"][0]
        is_cross = self.is_cross
        dim_head = extra_options["dim_head"]
        attn_heads = heads = extra_options["n_heads"]
        attn_precision = extra_options["attn_precision"]
        upcast_attention = attn_precision == torch.float32 and q.dtype != torch.float32
        self.upcast_attention = self.upcast_softmax = upcast_attention
        self.scale = dim_head**-0.5
        original_shape = extra_options["original_shape"]
        b, area, inner_dim = q.shape
        should_count = True #original_shape[0] == 2

        height_orig, width_orig = original_shape[-2:]
        aspect_ratio = width_orig / height_orig
        height = round((area / aspect_ratio)**0.5)
        width = round((area * aspect_ratio)**0.5)
        h=height
        if not hasattr(self.attention_store, 'H'): self.attention_store.H=0
        H_ = self.attention_store.H 
        if H_  == 0:
            self.attention_store.__init__()
        self.attention_store.H = max(H_, h) # hack
        rr=self.attention_store.H/h
        rr_i = int(rr)
        if rr_i % 2 == 0:
            r = rr_i
        else:
            r = round(rr)
        if self.manual_r is not None:
            r = self.manual_r
        # print('=== r',r,'height',height,'width',width, 'is_cross',is_cross,extra_options)
        # attention_store.between_steps()

        hidden_states = None
        # if not should_count:
        #     return optimized_attention_masked(q, k, v, heads, mask=mask, attn_precision=attn_precision)
        if not r >= 2:
            hidden_states = optimized_attention_masked(q, k, v, heads, mask=mask, attn_precision=attn_precision)
            # if self.attention_store.cur_att_layer >= self.attention_store.num_uncond_att_layers: ...
            #     # self.forward(attn, is_cross, place_in_unet, manual_r)
            # self.attention_store.cur_att_layer += 1
            # # reach the end
            # if self.attention_store.cur_att_layer == self.attention_store.num_att_layers + self.attention_store.num_uncond_att_layers:
            #     self.attention_store.cur_att_layer = 0
            #     self.attention_store.cur_step += 1
            #     self.attention_store.between_steps()
            # return hidden_states

        if r >= 2:
            attention_probs = self.get_attention_scores(q, k,attn_heads, mask)
            attention_probs_ = attention_probs
            attention_probs = attention_probs.view(-1, attn_heads, *attention_probs.shape[-2:]).mean(1)
            self.attention_store(
                attention_probs,
                is_cross,
                place_in_unet,
                self.manual_r,
                r=r,
                count = should_count,
            )
        if (is_cross and extra_options['block'] == ('output', 11) and extra_options['block_index']==0) or (
            is_cross and extra_options['block'] == ('output', 5) and extra_options['block_index']==1):
            self.attention_store.between_steps()
            self.attention_store.H = 0
        if hidden_states is not None: return hidden_states

        (v,) = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
            .contiguous(),
            (v,),
        )
        hidden_states = torch.bmm(attention_probs_, v)
        hidden_states = (
            hidden_states.unsqueeze_(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )

        return hidden_states

    # Modified from diffusers.models.attention_processor.Attention
    def get_attention_scores(self, query, key, heads, attention_mask=None):
        b, _, dim_head = query.shape
        dim_head //= heads
        scale = dim_head ** -0.5

        query, key = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
            .contiguous(),
            (query, key),
        )

        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        oom = False
        if not oom:
            attention_scores = torch.baddbmm(
                baddbmm_input,
                query,
                key.transpose(-1, -2),
                beta=beta,
                alpha=scale,
            )
            del baddbmm_input
        else:
            for i in range(len(baddbmm_input)):
                baddbmm_input[i:i+1] = torch.baddbmm(baddbmm_input[i:i+1],query[i:i+1],key.transpose(-1, -2)[i:i+1],beta=beta,alpha=scale)
                attention_scores = baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        if not oom:
            attention_probs = attention_scores.softmax(dim=-1)
            del attention_scores
        else:
            for i in range(len(attention_scores)):
                attention_scores[i:i+1] = attention_scores[i:i+1].softmax(dim=-1)
            attention_probs = attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

def get_mask(attention_store: AttentionStore,r: int=4, cond=None):
    """ Aggregates the attention across the different layers and heads at the specified resolution. """

    key_cross = f"r{r}_cross"
    key_self = f"r{r}_self"
    curr_r = r

    r_r = 1
    new_ca = 0
    new_fore=0
    a_n=0
    attention_maps = attention_store.get_average_attention()
    if len(attention_maps) == 0:
        attention_store.between_steps()
        attention_maps = attention_store.get_average_attention()

    while curr_r<=8:
        key_cross = f"r{curr_r}_cross"
        key_self = f"r{curr_r}_self"

        am_sa = attention_maps.get(key_self, [])
        am_ca = attention_maps.get(key_cross, [])
        if len(am_sa) == 0 or len(am_ca) == 0:
            if len(am_sa) == 0 and len(am_ca) == 0:
                print(f'[S-CFG] An error occured trying to get the attention masks. {key_self} {key_cross} Length:', len(am_sa), len(am_ca))
                # import pdb;pdb.set_trace()
            curr_r = int(curr_r * 2)
            r_r *= 2
            continue
        # elif len(am_sa) == 0:
        #     am_sa = am_ca
        # elif len(am_ca) == 0:
        #     am_ca = am_sa
            
        sa = torch.stack(am_sa, dim=1)
        ca = torch.stack(am_ca, dim=1)

        attn_num = sa.size(1)
        # sa = rearrange(sa, 'b n h w -> (b n) h w')
        # ca = rearrange(ca, 'b n h w -> (b n) h w')
        sa = sa.view(mul(*sa.shape[:2]), *sa.shape[2:])
        ca = ca.view(mul(*ca.shape[:2]), *ca.shape[2:])

        if sa.shape[0] < ca.shape[0]:
            sa=rep(sa,ca)
        elif ca.shape[0] < sa.shape[0]:
            ca=rep(ca,sa)
        
        

        curr = 0 # b hw c=hw
        curr +=sa
        ssgc_sa = curr
        ssgc_n =4
        for _ in range(ssgc_n-1):
            curr = sa@sa
            ssgc_sa += curr
        ssgc_sa/=ssgc_n
        sa = ssgc_sa
        ########smoothing ca
        ca_ = sa@ca # b hw c
        del ca
        torch.cuda.empty_cache()
        ca=ca_

        area = ca.size(1)
        height, width = cond.shape[-2:]
        aspect_ratio = width / height
        if aspect_ratio >= 1.0:
            h = round((area / aspect_ratio) ** 0.5)
            hw = (h, -1)
        else:
            w = round((area * aspect_ratio) ** 0.5)
            hw = (-1, w)


        # ca = rearrange(ca, 'b (h w) c -> b c h w', h=h )
        ca = ca.view(ca.size(0), *hw, ca.size(2)).permute(0, 3, 1, 2)
        if r_r>1:
            mode =  'bilinear' # 'nearest'
            # ca = F.interpolate(ca, scale_factor=r_r, mode=mode) # b 77 32 32
            ca = F.interpolate(ca, size=new_ca.shape[-2:], mode=mode)

        #####Gaussian Smoothing
        smoothing = getattr(attention_store, 'smoothing', None)
        if smoothing is None:
            kernel_size = 3
            sigma = 0.5
            smoothing = attention_store.smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(ca.device)
        channel = ca.size(1)
        # ca= rearrange(ca, ' b c h w -> (b c) h w' ).unsqueeze(1)
        ca = ca.contiguous().view(ca.size(0) * ca.size(1), *ca.shape[-2:]).unsqueeze_(1)
        ca = F.pad(ca, (1, 1, 1, 1), mode='reflect')
        ca = smoothing(ca.float()).squeeze(1)
        # ca = rearrange(ca, ' (b c) h w -> b c h w' , c= channel)
        ca = ca.view(-1, channel, *ca.shape[-2:])
        
        ca_norm = ca/(ca.mean(dim=[2,3], keepdim=True)+1e-8) ### spatial  normlization 
        # new_ca+=rearrange(ca_norm, '(b n) c h w -> b n c h w', n=attn_num).sum(1) 
        new_ca+=ca_norm.view(-1, attn_num, *ca_norm.shape[-3:]).sum(1) 

        fore_ca = torch.stack([ca[:,0],ca[:,1:].sum(dim=1)], dim=1)
        froe_ca_norm = fore_ca/fore_ca.mean(dim=[2,3], keepdim=True) ### spatial  normlization 
        # new_fore += rearrange(froe_ca_norm, '(b n) c h w -> b n c h w', n=attn_num).sum(1)  
        new_fore += froe_ca_norm.view(-1, attn_num, *froe_ca_norm.shape[-3:]).sum(1) 
        a_n+=attn_num

        curr_r = int(curr_r*2)
        r_r*=2
    
    new_ca = new_ca/a_n
    new_fore = new_fore/a_n
    new_ca   = new_ca.chunk(2, dim=0)[-1] #[1]
    fore_ca = new_fore.chunk(2, dim=0)[0]


    max_ca, inds = torch.max(new_ca[:,:], dim=1) 
    max_ca = max_ca.unsqueeze(1) # 
    ca_mask = (new_ca==max_ca).float() # b 77/10 16 16 


    max_fore, inds = torch.max(fore_ca[:,:], dim=1) 
    max_fore = max_fore.unsqueeze(1) # 
    fore_mask = (fore_ca==max_fore).float() # b 77/10 16 16 
    fore_mask = 1.0-fore_mask[:,:1] # b 1 16 16


    return [ ca_mask, fore_mask]


class SCFG:
    def __init__(self, model: ModelPatcher):
        self.model = model
        self.attention_store = AttentionStore()

    def _register_attention_control(self):
        number=0
        for name, module in self.model.model.diffusion_model.named_modules():
            if module.__class__.__name__ == "CrossAttention":
                parts = name.split(".")
                name = parts[-1]
                block_name = parts[0].split("_")[0]
                block_id = int(parts[1]) - (1 if block_name == "middle" else 0)
                block_index = int(parts[-2])
                # print((name, block_name, block_id, block_index)) # attn1, "input", 7, 0
                self.model.set_model_patch_replace(MyCrossAttnProcessor(self.attention_store, name == "attn2"), name, block_name, block_id, block_index)
                number+=1
        self.attention_store.num_att_layers = number

    def register_attention_control(self):
        # return self.register_attention_control2()
        number = 0
        for name, module in self.model.model.diffusion_model.named_modules():
            if module.__class__.__name__ == "CrossAttention":
                number +=1
        is_sdxl = number > 32
        number = 0
        m = 2 # multipler to also count the attn2 (cross-attention) blocks
        def select_blocks(a, down=True):
            b=a[-int(len(a)*(1/3)):] if down else a[:int(len(a)*(1/3))+1]
            return b
        # Patch for down blocks
        if not is_sdxl:
            for i in [1, 2, 4, 5, 7, 8]:
                # b=select_blocks([1, 2, 4, 5, 7, 8], True)
                # manual_r=4 if i in b else None
                manual_r=None
                self.model.set_model_attn1_replace(MyCrossAttnProcessor(self.attention_store, False, manual_r), "input", i)
                self.model.set_model_attn2_replace(MyCrossAttnProcessor(self.attention_store, True, manual_r), "input", i)
                number += 1 * m

            # Patch for up blocks
            for i in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
                # b=select_blocks([3, 4, 5, 6, 7, 8, 9, 10, 11], False)
                # manual_r=4 if i in b else None
                manual_r=None
                self.model.set_model_attn1_replace(MyCrossAttnProcessor(self.attention_store, False, manual_r), "output", i)
                self.model.set_model_attn2_replace(MyCrossAttnProcessor(self.attention_store, True, manual_r), "output", i)
                number += 1 * m

            # Patch for middle block
            # manual_r=8
            manual_r=None
            self.model.set_model_attn1_replace(MyCrossAttnProcessor(self.attention_store, False,  manual_r), "middle", 0)
            self.model.set_model_attn2_replace(MyCrossAttnProcessor(self.attention_store, True, manual_r), "middle", 0)
            number += 1 * m
        else:
            for id in [4,5,7,8]: # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    # Manually determine the r_cross/r_self for SDXL
                    # Taken from my observations of the selected blocks in SD1.x -> The third of the end of the input (down) blocks
                    b=select_blocks([4,5,7,8], True)
                    manual_r=4 if id in b else None
                    self.model.set_model_attn1_replace(MyCrossAttnProcessor(self.attention_store, False, manual_r), "input", id, index)
                    self.model.set_model_attn2_replace(MyCrossAttnProcessor(self.attention_store, True, manual_r), "input", id, index)
                    number += 1 * m
            for id in range(6): # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    # Manually determine the r_cross/r_self for SDXL
                    # Taken from my observations of the selected blocks in SD1.x -> The third of the beginning of the output (up) blocks
                    b=select_blocks(list(range(6)), False)
                    manual_r=4 if id in b else None
                    self.model.set_model_attn1_replace(MyCrossAttnProcessor(self.attention_store, False, manual_r), "output", id, index)
                    self.model.set_model_attn2_replace(MyCrossAttnProcessor(self.attention_store, True, manual_r), "output", id, index)
                    number += 1 * m
            for index in range(10):
                # Manually set the middle block to r8
                self.model.set_model_attn1_replace(MyCrossAttnProcessor(self.attention_store, False, manual_r=8), "middle", 0, index)
                self.model.set_model_attn2_replace(MyCrossAttnProcessor(self.attention_store, True, manual_r=8), "middle", 0, index)
                number += 1 * m
            
        self.attention_store.num_att_layers = number

    def apply_scfg(self, args):
        uncond = args["uncond"]
        cond = args["cond"]
        guidance_scale=args['cond_scale']
        noise_pred_uncond = uncond
        noise_pred_text = cond

        R = 4 # starting number when processing r#_cross/r#_self blocks
        
        ca_mask, fore_mask = get_mask(self.attention_store, r=R, cond=cond)

        if ca_mask is None or fore_mask is None:
            return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        mode = "nearest"
        # mode = "nearest-exact"
        # mask_t = F.interpolate(ca_mask, scale_factor=R, mode=mode)
        # mask_fore = F.interpolate(fore_mask, scale_factor=R, mode=mode)
        mask_t = F.interpolate(ca_mask, size=cond.shape[-2:], mode=mode)
        mask_fore = F.interpolate(fore_mask, size=cond.shape[-2:], mode=mode)

        ###eps
        model_delta = (noise_pred_text - noise_pred_uncond)
        model_delta_norm = model_delta.norm(dim=1, keepdim=True) # b 1 64 64

        delta_mask_norms = (model_delta_norm*mask_t).sum([2,3])/(mask_t.sum([2,3])+1e-8) # b 77
        upnormmax = delta_mask_norms.max(dim=1)[0] # b
        upnormmax = upnormmax.unsqueeze(-1)

        fore_norms = (model_delta_norm*mask_fore).sum([2,3])/(mask_fore.sum([2,3])+1e-8) # b 1

        up = fore_norms
        down = delta_mask_norms

        tmp_mask = (mask_t.sum([2,3])>0).float()
        rate = up*(tmp_mask)/(down+1e-8) # b 257
        rate = (rate.unsqueeze(-1).unsqueeze(-1)*mask_t).sum(dim=1, keepdim=True) # b 1, 64 64

        rate = torch.clamp(rate,min=0.8, max=3.0)
        rate = torch.clamp_max(rate, 15.0/guidance_scale)

        ###Gaussian Smoothing 
        kernel_size = 3
        sigma=0.5
        smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).to(rate.device)
        rate = F.pad(rate, (1, 1, 1, 1), mode='reflect')
        rate = smoothing(rate)


        rate = rate.to(noise_pred_text.dtype)

        noise_pred = noise_pred_uncond + guidance_scale * rate * (noise_pred_text - noise_pred_uncond)

        return noise_pred

class SemanticAwareGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/cfg"
    instances = WeakSet()

    @classmethod
    def IS_CHANGED(s, *args, **kwargs):
        for o in s.instances:
            if hasattr(o, 'scfg'):
                o.scfg.attention_store.__init__()
        return ""
    
    def __init__(self) -> None:
        self.__class__.instances.add(self)
    
    def patch(self, model:ModelPatcher):
        model = model.clone()
        scfg = SCFG(model)
        scfg.register_attention_control()
        scfg.attention_store.reset()
        scfg.model.set_model_sampler_cfg_function(scfg.apply_scfg)
        self.scfg=scfg
        return (scfg.model,)


NODE_CLASS_MAPPINGS = {
    "SemanticAwareGuidance": SemanticAwareGuidance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SemanticAwareGuidance": "Semantic-aware Guidance",
}
