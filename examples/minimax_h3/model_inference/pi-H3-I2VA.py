import torch
from PIL import Image, ImageDraw
from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Pipeline, ModelConfig
from diffsynth.utils.data.audio_video import write_video_audio

num_inference_steps = 4
use_dwd = True
adapter_name = f"pi_h3_{num_inference_steps}step" + ("_dwd" if use_dwd else "") + "_adapter"

model_configs = [
    ModelConfig(model_id="multimodalart/MiniMax-H3-Pruned", origin_file_pattern="transformer/diffusion_pytorch_model-*.safetensors", download_source="huggingface"),
    ModelConfig(model_id="multimodalart/MiniMax-H3-Pruned", origin_file_pattern="text_encoder/model-*.safetensors", download_source="huggingface"),
    ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/audio_vae/model.safetensors"),
]
model_configs.append(ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/video_vae/source/model.safetensors"))

pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    pi_h3_adapter_config=ModelConfig(model_id="Lakonik/pi-H3", origin_file_pattern=f"{adapter_name}/diffusion_pytorch_model.safetensors"),
    model_configs=model_configs,
    processor_config=ModelConfig(model_id="MiniMax/MiniMax-H3", origin_file_pattern="FL2VA/processor/"),
)

keyframes = []
for left, right in [(64, 128)]:
    image = Image.new("RGB", (384, 256), "#172846")
    ImageDraw.Draw(image).ellipse((left, 96, right, 160), fill="#F0AC46")
    keyframes.append(image)

video, audio = pipe(
    prompt="A gold ball rolls from left to right across a blue stage, with a quiet rolling sound.",
    height=256, width=384, num_frames=124,
    seed=42, num_inference_steps=num_inference_steps,
    keyframes=keyframes, keyframe_indices=[0],
)
write_video_audio(video, audio, 'pi-h3-i2va.mp4', fps=24, audio_sample_rate=32000)
