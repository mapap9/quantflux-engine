import torch
import time
import gc
from diffusers import Flux2KleinPipeline
import os
import sys

# Calculate the absolute path to the repository root (one level up from benchmarks/)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Append it to the system path if it isn't already tracked
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.patcher import apply_runtime_system_patch
from src.wrappers import optimize_production_flux_graph

def flush_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def run_production_suite():
    print("🚀 Initializing Frontier FLUX.2 [klein] 4B Out-of-Core Optimization Benchmark...")
    
    if not torch.cuda.is_available():
        print("❌ CRITICAL: This benchmark requires a high-performance hardware accelerator.")
        return

    model_id = "black-forest-labs/FLUX.2-klein-4B"
    device = "cuda"
    dtype = torch.bfloat16

    # 1. Load Pipeline STRICTORLY ON CPU to protect VRAM boundaries
    print(f"⏳ Downloading pipeline to host memory (CPU): {model_id}...")
    try:
        pipeline = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        return

    # 2. Compute Text Embeddings on CPU
    print("⏳ Processing text conditioning layers on CPU...")
    prompt = "A high-fidelity hardware circuit layout of a custom AI accelerator chip."
    batch_size = 1

    with torch.no_grad():
        # Cleanly unpack the 2 tensors returned by the underlying pipeline code
        prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt, device="cpu", num_images_per_prompt=batch_size
        )

        # Pull or simulate the pooled representation matching the batch profile
        # Flux2-Klein typically derives pooled contexts dynamically or uses mean pooling
        # across the sequence dimension for its conditioning projections.
        pooled_prompt_embeds = prompt_embeds.mean(dim=1)

    # Push ONLY the target layout vectors to the active device subsystem
    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    text_ids = text_ids.to(device)

    # 3. Extract the visual transformer and purge the remaining CPU pipelines from memory
    transformer = pipeline.transformer
    del pipeline  # Completely delete text encoders/tokenizers to free host memory
    gc.collect()

    # 4. Push the raw transformer to GPU for Baseline Profiling
    print("⏳ Staging visual transformer to GPU...")
    transformer = transformer.to(device)
    
    latents = torch.randn(batch_size, 4096, 128, device=device, dtype=dtype)
    latent_ids = torch.zeros(batch_size, 4096, 4, device=device, dtype=dtype)
    guidance = torch.tensor([3.5], device=device, dtype=dtype)
    timesteps = torch.tensor([500], device=device, dtype=dtype)

    # --- BASELINE INFERENCE RUN ---
    print("\n📊 RUN 1: Profiling Unoptimized Production Baseline...")
    transformer.eval()
    flush_memory()
    
    try:
        t0 = time.time()
        with torch.no_grad():
            _ = transformer(
                hidden_states=latents,
                timestep=timesteps,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids
            )
        torch.cuda.synchronize()
        base_latency = (time.time() - t0) * 1000
        base_vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"❌ Baseline Peak VRAM: {base_vram:.2f} GB | Latency: {base_latency:.2f}ms")
    except torch.cuda.OutOfMemoryError:
        print("❌ Baseline triggered an expected OutOfMemoryError on your 16GB hardware configuration!")
        base_vram = 16.0  # Cap approximation for comparison metrics

    # --- MONKEY-PATCH OVERRIDE INJECTION ---
    print("\n⚡ RUN 2: Deploying Low-Level Runtime Systems Patch Override...")
    transformer.train() 
    
    # Intercept and modify the live model object
    transformer = optimize_production_flux_graph(transformer)
    flush_memory()

    t1 = time.time()
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        out = transformer(
            hidden_states=latents,
            timestep=timesteps,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids
        )
        loss = out.sample.mean()
        loss.backward()
        
    torch.cuda.synchronize()
    opt_vram = torch.cuda.max_memory_allocated() / 1e9
    opt_latency = (time.time() - t1) * 1000
    
    vram_saved = max(0.0, base_vram - opt_vram)
    percent_saved = (vram_saved / base_vram) * 100

    print("\n📊 FLUX.2 PRODUCTION TELEMETRY ANALYSIS REPORT")
    print("=" * 55)
    if base_vram == 16.0:
        print(f"❌ Unoptimized Baseline Footprint : > 16.00 GB (OOM Limit)")
    else:
        print(f"❌ Unoptimized Baseline Footprint : {base_vram:.2f} GB")
    print(f"✅ Patched Runtime Engine Footprint : {opt_vram:.2f} GB")
    print("-" * 55)
    print(f"🔥 Hardware Parameter VRAM Saved : {vram_saved:.2f} GB ({percent_saved:.1f}% Reduction)")
    print(f"⏱️ Optimization Step Execution Latency: {opt_latency:.2f}ms")
    print("=" * 55)

if __name__ == "__main__":
    run_production_suite()
