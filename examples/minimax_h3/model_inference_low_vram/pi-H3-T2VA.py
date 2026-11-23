import torch
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio

num_inference_steps = 4
use_dwd = True
adapter_name = f"pi_h3_{num_inference_steps}step" + ("_dwd" if use_dwd else "") + "_adapter"

vram_config = {
    "offload_dtype": torch.bfloat16, "offload_device": "cpu",
    "onload_dtype": torch.bfloat16, "onload_device": "cpu",
    "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
    "computation_dtype": torch.bfloat16, "computation_device": "cuda",
}
model_configs = [
    ModelConfig(model_id="multimodalart/MiniMax-H3-Pruned", origin_file_pattern="transformer/diffusion_pytorch_model-*.safetensors", download_source="huggingface", **vram_config),
    ModelConfig(model_id="multimodalart/MiniMax-H3-Pruned", origin_file_pattern="text_encoder/model-*.safetensors", download_source="huggingface", **vram_config),
    ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors", **vram_config),
]
if not use_dwd:
    model_configs.append(ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors", **vram_config))

pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    pi_h3_adapter_config=ModelConfig(model_id="Lakonik/pi-H3", origin_file_pattern=f"{adapter_name}/diffusion_pytorch_model.safetensors"),
    model_configs=model_configs,
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)

video, audio = pipe(
    prompt="A gold ball rolls from left to right across a blue stage, with a quiet rolling sound.",
    height=256, width=384, num_frames=124,
    seed=42, num_inference_steps=num_inference_steps,
)
write_video_audio(video, audio, 'pi-h3-t2va-low-vram.mp4', fps=24, audio_sample_rate=32000)
