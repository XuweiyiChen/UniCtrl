import os
import torch

# fix all the seeds for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
import ptp_utils
import random
import abc
import gradio as gr
from glob import glob
from einops import rearrange
from omegaconf import OmegaConf
from safetensors import safe_open
from diffusers import AutoencoderKL
from diffusers import DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.pipelines.pipeline_animation import AnimationCtrlPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import (
    convert_ldm_unet_checkpoint,
    convert_ldm_clip_checkpoint,
)
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    convert_ldm_vae_checkpoint,
)
from diffusers.training_utils import set_seed


pretrained_model_path = "./models/StableDiffusion"
inference_config_path = "configs/inference/inference-v1.yaml"

css = """
.toolbutton {
    margin-buttom: 0em 0em 0em 0em;
    max-width: 2.5em;
    min-width: 2.5em !important;
    height: 2.5em;
}
"""


class AttentionControl(abc.ABC):
    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, hidden_states, video_length, place_in_unet: str):
        hidden_states = rearrange(hidden_states, "(b f) d c -> b f d c", f=video_length)
        batch_size = hidden_states.shape[0] // 2

        if batch_size == 2:
            # Do classifier-free guidance
            hidden_states_uncondition, hidden_states_condition = hidden_states.chunk(2)

            if self.cur_step <= self.motion_control_step:
                hidden_states_motion_uncondition = hidden_states_uncondition[
                    1
                ].unsqueeze(0)
            else:
                hidden_states_motion_uncondition = hidden_states_uncondition[
                    0
                ].unsqueeze(0)

            hidden_states_out_uncondition = torch.cat(
                [
                    hidden_states_motion_uncondition,
                    hidden_states_uncondition[1].unsqueeze(0),
                ],
                dim=0,
            )  # Query
            hidden_states_sac_in_uncondition = self.forward(
                hidden_states_uncondition[0].unsqueeze(0), video_length, place_in_unet
            )
            hidden_states_sac_out_uncondition = torch.cat(
                [
                    hidden_states_sac_in_uncondition,
                    hidden_states_uncondition[1].unsqueeze(0),
                ],
                dim=0,
            )  # Key & Value

            if self.cur_step <= self.motion_control_step:
                hidden_states_motion_condition = hidden_states_condition[1].unsqueeze(0)
            else:
                hidden_states_motion_condition = hidden_states_condition[0].unsqueeze(0)

            hidden_states_out_condition = torch.cat(
                [
                    hidden_states_motion_condition,
                    hidden_states_condition[1].unsqueeze(0),
                ],
                dim=0,
            )  # Query
            hidden_states_sac_in_condition = self.forward(
                hidden_states_condition[0].unsqueeze(0), video_length, place_in_unet
            )
            hidden_states_sac_out_condition = torch.cat(
                [
                    hidden_states_sac_in_condition,
                    hidden_states_condition[1].unsqueeze(0),
                ],
                dim=0,
            )  # Key & Value

            hidden_states_out = torch.cat(
                [hidden_states_out_uncondition, hidden_states_out_condition], dim=0
            )
            hidden_states_sac_out = torch.cat(
                [hidden_states_sac_out_uncondition, hidden_states_sac_out_condition],
                dim=0,
            )

        elif batch_size == 1:
            if self.cur_step <= self.motion_control_step:
                hidden_states_motion = hidden_states[1].unsqueeze(0)
            else:
                hidden_states_motion = hidden_states[0].unsqueeze(0)

            hidden_states_out = torch.cat(
                [hidden_states_motion, hidden_states[1].unsqueeze(0)], dim=0
            )  # Query
            hidden_states_sac_in = self.forward(
                hidden_states[0].unsqueeze(0), video_length, place_in_unet
            )
            hidden_states_sac_out = torch.cat(
                [hidden_states_sac_in, hidden_states[1].unsqueeze(0)], dim=0
            )  # Key & Value

        else:
            raise gr.Error(f"Not implemented error")
        hidden_states = rearrange(hidden_states, "b f d c -> (b f) d c", f=video_length)
        hidden_states_out = rearrange(
            hidden_states_out, "b f d c -> (b f) d c", f=video_length
        )
        hidden_states_sac_out = rearrange(
            hidden_states_sac_out, "b f d c -> (b f) d c", f=video_length
        )
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
        return hidden_states_out, hidden_states_sac_out, hidden_states_sac_out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_att_layers = -1
        self.motion_control_step = 0

    def __init__(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_att_layers = -1
        self.motion_control_step = 0


class EmptyControl(AttentionControl):
    def forward(self, hidden_states, video_length, place_in_unet):
        return hidden_states


class FreeSAC(AttentionControl):
    def forward(self, hidden_states, video_length, place_in_unet):
        hidden_states_sac = (
            hidden_states[:, 0, :, :].unsqueeze(1).repeat(1, video_length, 1, 1)
        )
        return hidden_states_sac


examples = [
    # 0-RealisticVision
    [
        "realisticVisionV60B1_v20Novae.safetensors",
        "mm_sd_v14.ckpt",
        "A panda standing on a surfboard in the ocean under moonlight.",
        "worst quality, low quality, nsfw, logo",
        0.2,
        512,
        512,
        "12345",
        ["use_fp16"],
    ],
    # [
    #     "toonyou_beta3.safetensors",
    #     "mm_sd_v14.ckpt",
    #     "(best quality, masterpiece), 1girl, looking at viewer, blurry background, upper body, contemporary, dress",
    #     "(worst quality, low quality)",
    #     0.2,
    #     512,
    #     512,
    #     "12345",
    #     ["use_fp16"],
    # ],
    # [
    #     "lyriel_v16.safetensors",
    #     "mm_sd_v14.ckpt",
    #     "hypercars cyberpunk moving, muted colors, swirling color smokes, legend, cityscape, space",
    #     "3d, cartoon, anime, sketches, worst quality, low quality, nsfw, logo",
    #     0.2,
    #     512,
    #     512,
    #     "12345",
    #     ["use_fp16"],
    # ],
    # [
    #     "rcnzCartoon3d_v10.safetensors",
    #     "mm_sd_v14.ckpt",
    #     "A cute raccoon playing guitar in a boat on the ocean",
    #     "worst quality, low quality, nsfw, logo",
    #     0.2,
    #     512,
    #     512,
    #     "42",
    #     ["use_fp16"],
    # ],
    # [
    #     "majicmixRealistic_v5Preview.safetensors",
    #     "mm_sd_v14.ckpt",
    #     "1girl, reading book",
    #     "(ng_deepnegative_v1_75t:1.2), (badhandv4:1), (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, bad hands, watermark, moles",
    #     0.2,
    #     512,
    #     512,
    #     "12345",
    #     ["use_fp16"],
    # ],
]

# clean Gradio cache
print(f"### Cleaning cached examples ...")
os.system(f"rm -rf gradio_cached_examples/")


class AnimateController:
    def __init__(self):
        # config dirs
        self.basedir = os.getcwd()
        self.stable_diffusion_dir = os.path.join(
            self.basedir, "models", "StableDiffusion"
        )
        self.motion_module_dir = os.path.join(self.basedir, "models", "Motion_Module")
        self.personalized_model_dir = os.path.join(
            self.basedir, "models", "DreamBooth_LoRA"
        )
        self.savedir = os.path.join(self.basedir, "samples")
        os.makedirs(self.savedir, exist_ok=True)

        self.base_model_list = [None]
        self.motion_module_list = []
        self.selected_base_model = None
        self.selected_motion_module = None
        self.set_width = None
        self.set_height = None

        self.refresh_motion_module()
        self.refresh_personalized_model()

        # config models
        self.inference_config = OmegaConf.load(inference_config_path)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            pretrained_model_path, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_path, subfolder="text_encoder"
        ).cuda()
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_path, subfolder="vae"
        ).cuda()
        self.unet = UNet3DConditionModel.from_pretrained_2d(
            pretrained_model_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                self.inference_config.unet_additional_kwargs
            ),
        ).cuda()

        self.freq_filter = None

        self.update_base_model(self.base_model_list[-2])
        self.update_motion_module(self.motion_module_list[0])

    def refresh_motion_module(self):
        motion_module_list = glob(os.path.join(self.motion_module_dir, "*.ckpt"))
        self.motion_module_list = sorted(
            [os.path.basename(p) for p in motion_module_list]
        )

    def refresh_personalized_model(self):
        base_model_list = glob(
            os.path.join(self.personalized_model_dir, "*.safetensors")
        )
        self.base_model_list += sorted([os.path.basename(p) for p in base_model_list])

    def update_base_model(self, base_model_dropdown):
        self.selected_base_model = base_model_dropdown
        if base_model_dropdown == "None" or base_model_dropdown is None:
            return gr.Dropdown.update()

        base_model_dropdown = os.path.join(
            self.personalized_model_dir, base_model_dropdown
        )
        base_model_state_dict = {}
        with safe_open(base_model_dropdown, framework="pt", device="cpu") as f:
            for key in f.keys():
                base_model_state_dict[key] = f.get_tensor(key)

        converted_vae_checkpoint = convert_ldm_vae_checkpoint(
            base_model_state_dict, self.vae.config
        )
        self.vae.load_state_dict(converted_vae_checkpoint)

        converted_unet_checkpoint = convert_ldm_unet_checkpoint(
            base_model_state_dict, self.unet.config
        )
        self.unet.load_state_dict(converted_unet_checkpoint, strict=False)

        self.text_encoder = convert_ldm_clip_checkpoint(base_model_state_dict)
        return gr.Dropdown.update()

    def update_motion_module(self, motion_module_dropdown):
        self.selected_motion_module = motion_module_dropdown

        motion_module_dropdown = os.path.join(
            self.motion_module_dir, motion_module_dropdown
        )
        motion_module_state_dict = torch.load(
            motion_module_dropdown, map_location="cpu"
        )
        _, unexpected = self.unet.load_state_dict(
            motion_module_state_dict, strict=False
        )
        assert len(unexpected) == 0
        return gr.Dropdown.update()

    def run_pipeline(self, pipeline, args):
        # Initialize CUDA context in the subprocess
        torch.cuda.init()
        # Run the pipeline with the given arguments
        return pipeline(**args)

    def animate_ctrl(
        self,
        base_model_dropdown,
        motion_module_dropdown,
        prompt_textbox,
        negative_prompt_textbox,
        motion_control,
        width_slider,
        height_slider,
        seed_textbox,
        # speed up
        speed_up_options,
    ):
        set_seed(42)
        inference_step = 25

        if self.selected_base_model != base_model_dropdown:
            self.update_base_model(base_model_dropdown)
        if self.selected_motion_module != motion_module_dropdown:
            self.update_motion_module(motion_module_dropdown)

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()

        if int(seed_textbox) > 0:
            seed = int(seed_textbox)
        else:
            seed = random.randint(1, 1e16)
        torch.manual_seed(int(seed))

        assert seed == torch.initial_seed()
        print(f"### seed: {seed}")

        generator = torch.Generator(device="cuda:0")
        generator.manual_seed(seed)

        pipeline = AnimationCtrlPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=DDIMScheduler(
                **OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs)
            ),
        ).to("cuda")

        motion_control_step = motion_control * inference_step

        attn_controller = FreeSAC()
        attn_controller.motion_control_step = motion_control_step
        ptp_utils.register_attention_control(pipeline, attn_controller)

        sample_output_ctrl = pipeline(
            prompt_textbox,
            negative_prompt=negative_prompt_textbox,
            num_inference_steps=inference_step,
            guidance_scale=7.5,
            width=width_slider,
            height=height_slider,
            video_length=16,
            use_fp16=True if "use_fp16" in speed_up_options else False,
            generator=generator,
        )

        ctrl_sample = sample_output_ctrl.videos

        save_ctrl_sample_path = os.path.join(self.savedir, "ctrl_sample.mp4")
        save_videos_grid(ctrl_sample, save_ctrl_sample_path)

        json_config = {
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "width": width_slider,
            "height": height_slider,
            "seed": seed,
            "base_model": base_model_dropdown,
            "motion_module": motion_module_dropdown,
            "use_fp16": True if "use_fp16" in speed_up_options else False,
        }

        del attn_controller
        del pipeline
        torch.cuda.empty_cache()
        return (
            gr.Video.update(value=save_ctrl_sample_path),
            gr.Json.update(value=json_config),
        )

    def animate(
        self,
        base_model_dropdown,
        motion_module_dropdown,
        prompt_textbox,
        negative_prompt_textbox,
        motion_control,
        width_slider,
        height_slider,
        seed_textbox,
        # freeinit params
        filter_type_dropdown,
        speed_up_options,
    ):
        # set global seed
        set_seed(42)
        # set inference step
        inference_step = 25

        if self.selected_base_model != base_model_dropdown:
            self.update_base_model(base_model_dropdown)
        if self.selected_motion_module != motion_module_dropdown:
            self.update_motion_module(motion_module_dropdown)

        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()

        if seed_textbox and int(seed_textbox) >= 0:
            seed = int(seed_textbox)
        else:
            seed = random.randint(0, 2**32 - 1)
        torch.manual_seed(int(seed))

        assert seed == torch.initial_seed()
        print(f"seed: {seed}")

        generator = torch.Generator(device="cuda:0")
        generator.manual_seed(seed)
        
        pipeline = AnimationPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=DDIMScheduler(
                **OmegaConf.to_container(self.inference_config.noise_scheduler_kwargs)
            ),
        ).to("cuda")

        attn_controller = EmptyControl()
        attn_controller.motion_control_step = -1
        ptp_utils.register_attention_control(pipeline, attn_controller)

        sample_output_orig = pipeline(
            prompt_textbox,
            negative_prompt=negative_prompt_textbox,
            num_inference_steps=inference_step,
            guidance_scale=7.5,
            width=width_slider,
            height=height_slider,
            video_length=16,
            use_fp16=(
                True if speed_up_options and "use_fp16" in speed_up_options else False
            ),
            generator=generator,
        )

        orig_sample = sample_output_orig.videos

        save_orig_sample_path = os.path.join(self.savedir, "orig_sample.mp4")
        save_videos_grid(orig_sample, save_orig_sample_path)

        json_config = {
            "prompt": prompt_textbox,
            "n_prompt": negative_prompt_textbox,
            "width": width_slider,
            "height": height_slider,
            "seed": seed,
            "base_model": base_model_dropdown,
            "motion_module": motion_module_dropdown,
            "filter_type": filter_type_dropdown,
            "use_fp16": (
                True if speed_up_options and "use_fp16" in speed_up_options else False
            ),
        }
        del pipeline
        torch.cuda.empty_cache()

        return (
            gr.Video.update(value=save_orig_sample_path),
            gr.Json.update(value=json_config),
        )


controller = AnimateController()


def ui():
    with gr.Blocks(css=css) as demo:
        # gr.Markdown('# FreeInit')
        gr.Markdown(
            """
            <div align="center">
            <h1>UniCtrl: Improving the Spatiotemporal Consistency of Text-to-Video Diffusion Models via Training-Free Unified Attention Control</h1>
            </div>
            """
        )
        gr.Markdown(
            """
            <p align="center">
                    <a title="Project Page" href="https://unified-attention-control.github.io/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://img.shields.io/badge/Project-Website-5B7493?logo=googlechrome&logoColor=5B7493">
                    </a>
                    <a title="arXiv" href="https://arxiv.org/abs/2312.07537" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b">
                    </a>
                    <a title="GitHub" href="https://github.com/XuweiyiChen/UniCtrl" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
                        <img src="https://img.shields.io/github/stars/XuweiyiChen/UniCtrl?label=GitHub%E2%98%85&&logo=github" alt="badge-github-stars">
                    </a>
            </p>
            """
        )
        gr.Markdown(
            """
            Official Gradio Demo for ***UniCtrl: Improving the Spatiotemporal Consistency of Text-to-Video Diffusion Models via Training-Free Unified Attention Control***.
            UniCtrl improves spatiotemporal consistency of diffusion-based video generation at inference time. In this demo, we apply FreeInit on [AnimateDiff v1](https://github.com/guoyww/AnimateDiff) as an example. Sampling time: ~ 80s.<br>
            """
        )

        with gr.Row():
            with gr.Column():
                prompt_textbox = gr.Textbox(
                    label="Prompt", lines=3, placeholder="Enter your prompt here"
                )
                negative_prompt_textbox = gr.Textbox(
                    label="Negative Prompt",
                    lines=3,
                    value="worst quality, low quality, nsfw, logo",
                )
                motion_control = gr.Slider(
                    label="Motion Injection Degree",
                    value=0.2,
                    minimum=0,
                    maximum=1,
                    step=0.1,
                    info="Motion Control Strength",
                )

                gr.Markdown(
                    """
                    *Prompt Tips:*

                    For each personalized model in `Model Settings`, you can refer to their webpage on CivitAI to learn how to write good prompts for them:
                    - [`realisticVisionV60B1_v20Novae.safetensors.safetensors`](https://civitai.com/models/4201?modelVersionId=130072)
                    - [`toonyou_beta3.safetensors`](https://civitai.com/models/30240?modelVersionId=78775)
                    - [`lyriel_v16.safetensors`](https://civitai.com/models/22922/lyriel)
                    - [`rcnzCartoon3d_v10.safetensors`](https://civitai.com/models/66347?modelVersionId=71009)
                    - [`majicmixRealistic_v5Preview.safetensors`](https://civitai.com/models/43331?modelVersionId=79068)   
                    """
                )

                with gr.Accordion("Model Settings", open=False):
                    gr.Markdown(
                        """
                        Select personalized model and motion module for AnimateDiff.
                        """
                    )
                    base_model_dropdown = gr.Dropdown(
                        label="Base DreamBooth Model",
                        choices=controller.base_model_list,
                        value=controller.base_model_list[-2],
                        interactive=True,
                        info="Select personalized text-to-image model from community",
                    )
                    motion_module_dropdown = gr.Dropdown(
                        label="Motion Module",
                        choices=controller.motion_module_list,
                        value=controller.motion_module_list[0],
                        interactive=True,
                        info="Select motion module. Recommend mm_sd_v14.ckpt for larger movements.",
                    )

                base_model_dropdown.change(
                    fn=controller.update_base_model,
                    inputs=[base_model_dropdown],
                    outputs=[base_model_dropdown],
                )
                motion_module_dropdown.change(
                    fn=controller.update_motion_module,
                    inputs=[motion_module_dropdown],
                    outputs=[base_model_dropdown],
                )

                with gr.Accordion("Advance", open=False):
                    with gr.Row():
                        width_slider = gr.Slider(
                            label="Width", value=512, minimum=256, maximum=1024, step=64
                        )
                        height_slider = gr.Slider(
                            label="Height",
                            value=512,
                            minimum=256,
                            maximum=1024,
                            step=64,
                        )
                    with gr.Row():
                        seed_textbox = gr.Textbox(label="Seed", value=442)
                        seed_button = gr.Button(
                            value="\U0001F3B2", elem_classes="toolbutton"
                        )
                        seed_button.click(
                            fn=lambda: gr.Textbox.update(value=random.randint(1, 1e9)),
                            inputs=[],
                            outputs=[seed_textbox],
                        )
                    with gr.Row():
                        speed_up_options = gr.CheckboxGroup(
                            ["use_fp16"],
                            label="Speed-Up Options",
                            value=["use_fp16"],
                        )

            with gr.Column():
                with gr.Row():
                    orig_video = gr.Video(label="AnimateDiff", interactive=False)
                    ctrl_video = gr.Video(
                        label="AnimateDiff + UniCtrl", interactive=False
                    )
                with gr.Row():
                    generate_button = gr.Button(
                        value="Generate Original", variant="primary"
                    )
                    generate_button_ctr = gr.Button(
                        value="Generate UniCtrl", variant="primary"
                    )
                with gr.Row():
                    json_config = gr.Json(label="Config", value=None)

            inputs = [
                base_model_dropdown,
                motion_module_dropdown,
                prompt_textbox,
                negative_prompt_textbox,
                motion_control,
                width_slider,
                height_slider,
                seed_textbox,
                speed_up_options,
            ]

            generate_button.click(
                fn=controller.animate, inputs=inputs, outputs=[orig_video, json_config]
            )
            generate_button_ctr.click(
                fn=controller.animate_ctrl,
                inputs=inputs,
                outputs=[ctrl_video, json_config],
            )

        gr.Examples(
            fn=controller.animate_ctrl,
            examples=examples,
            inputs=inputs,
            outputs=[ctrl_video, json_config],
            cache_examples=True,
        )

    return demo


if __name__ == "__main__":
    demo = ui()
    demo.queue(max_size=20)
    demo.launch(server_name="localhost", share=True)
