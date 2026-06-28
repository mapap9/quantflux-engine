import torch
import time
import gc
import json
import os
from diffusers import Flux2KleinPipeline
from src.wrappers import optimize_production_flux_graph

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()

def run_final_pipeline():
    print("🚀 Initializing Optimized Production FLUX.2 Inference Engine...")

    if not torch.cuda.is_available():
        print("❌ CRITICAL: Hardware accelerator required.")
        return

    model_id = "black-forest-labs/FLUX.2-klein-4B"
    device = "cuda"
    dtype = torch.bfloat16
    prompt = "A high-fidelity hardware circuit layout of a custom AI accelerator chip, blueprint style, intricate details, neon blue traces, dark background."

    # 1. Load pipeline completely to CPU host memory
    print(f"⏳ Loading pipeline to CPU host memory...")
    pipeline = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)

# 2. Run Text Encoding on CPU BEFORE moving anything to GPU
    print("⏳ Encoding text conditioning inputs on host memory...")
    with torch.no_grad():
        enc_outputs = pipeline.encode_prompt(
            prompt=prompt, device="cpu", num_images_per_prompt=1
        )

        # Correct structural naming: the second tensor represents the image coordinate IDs
        if len(enc_outputs) == 2:
            prompt_embeds, img_ids = enc_outputs
            pooled_prompt_embeds = None
        else:
            prompt_embeds, pooled_prompt_embeds, img_ids = enc_outputs[:3]

    # 3. Evict Text Encoders immediately to lock in maximum VRAM headroom
    print("🧹 Offloading heavy text encoders from the pipeline block...")
    if hasattr(pipeline, "text_encoder"): del pipeline.text_encoder
    if hasattr(pipeline, "text_encoder_2"): del pipeline.text_encoder_2
    flush_memory()

    # 4. Inject runtime optimization patches onto the transformer graph while on CPU
    print("\n⚡ Injecting runtime optimization patches onto the transformer graph...")
    pipeline.transformer = optimize_production_flux_graph(pipeline.transformer)
    pipeline.transformer.eval()

    # 5. Safe Pipeline Migration to CUDA
    print("⏳ Staging runtime execution graph onto GPU memory...")
    pipeline = pipeline.to(device)
    flush_memory()

    # 6. Configure VAE memory optimization flags
    pipeline.vae.enable_slicing()
    pipeline.vae.enable_tiling()

    print("\n🏃 Executing production inference pass via optimized pipeline...")
    t0 = time.time()

    # Pack ONLY the standard framework signature arguments
    # The pipeline internally passes these to prepare_latents() to generate the 4D tracking matrices
    pipe_kwargs = {
        "prompt_embeds": prompt_embeds.to(device),
        "num_inference_steps": 20,
        "height": 512,
        "width": 512,
        "generator": torch.manual_seed(42)
    }
    if pooled_prompt_embeds is not None:
        pipe_kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds.to(device)

    # Run inference with all tracking layers unified on CUDA
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            image = pipeline(**pipe_kwargs).images[0]

    print(f"✅ Image generation completed in {(time.time() - t0):.2f}s.")

    # 7. Save the finalized asset to disk
    output_path = "output_patched.png"
    image.save(output_path)
    print(f"🎉 SUCCESS: Finalized asset generated successfully and saved to: '{output_path}'")

if __name__ == "__main__":
    run_final_pipeline()
