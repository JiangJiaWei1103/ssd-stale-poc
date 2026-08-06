"""Modal launcher: build an image with the LOCAL sglang branch (staleness knob)
+ this harness, then run all gates + the deliverable matrix in one launch on an
A10G. Results land on a Modal Volume.

    modal run modal/app.py

Recipe verified against the branch's pins (torch 2.11 / CUDA 13.0.1 / py3.12,
sgl-kernel==0.4.5 prebuilt wheel, flashinfer[cu13] supports sm_86) and Modal's
CUDA/local-source conventions. See docs/ for the flagged risks (sgl-kernel sm_86
SASS coverage is the #1 unknown -- there's a fail-fast import smoke test below).
"""
import modal

LOCAL_SGLANG = "/Users/abaowei/Desktop/Fun/my_sglang/dev/sglang"
LOCAL_HARNESS = "/Users/abaowei/Desktop/Fun/my_sglang/dev/ssd-stale-poc"
CUDA_TAG = "13.0.1-devel-ubuntu24.04"  # docker/Dockerfile ARG; devel = nvcc for triton/flashinfer JIT
TORCH_CU = "https://download.pytorch.org/whl/cu130"  # Dockerfile: 13.0.1 -> cu130
HF_CACHE = "/hf"
OUT_DIR = "/out"

hf_vol = modal.Volume.from_name("sglang-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("ssd-stale-results", create_if_missing=True)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .entrypoint([])  # clear the nvidia/cuda ENTRYPOINT (Modal CUDA guide)
    .apt_install("git", "build-essential", "cmake", "ninja-build", "libnuma-dev")
    .env({
        "SGLANG_BUILD_RUST_EXTS": "none",       # setup.py:180 -> skip Rust build (no cargo needed)
        "SGLANG_RAGGED_VERIFY_MODE": "static",  # pin verify-all
        "HF_HOME": HF_CACHE,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .pip_install("hf_transfer", "huggingface_hub", "datasets", "matplotlib")
    # ship the LOCAL branch (environ.py edit + dspark_hidden_lag_cache.py) INTO the image.
    # copy=True is REQUIRED so the pip install -e (a build step) sees these files.
    # remote_path must NOT be basename "sglang": /root is on sys.path (app.py lives
    # there), so a dir literally named /root/sglang would shadow the installed
    # package as an empty PEP-420 namespace package. Ship to /root/sglang_src.
    .add_local_dir(
        LOCAL_SGLANG, remote_path="/root/sglang_src", copy=True,
        ignore=[".git", ".git/**", "**/__pycache__/**", "**/*.pyc",
                "**/*.so", "**/*.dylib", "**/target/**", "**/node_modules/**",
                "docs/**", ".venv/**", "**/.pytest_cache/**"],
    )
    .run_commands(
        "python -m pip install --upgrade pip setuptools wheel",
        # sgl-kernel: prebuilt wheel, --no-deps (Dockerfile cu13 branch)
        "python -m pip install --no-deps sglang-kernel==0.4.5",
        # editable install of the branch; cu130 extra-index supplies torch 2.11 for CUDA 13
        f'python -m pip install --extra-index-url {TORCH_CU} -e "/root/sglang_src/python"',
    )
    # ship the harness (importable at runtime; not needed at build time -> copy=False)
    .add_local_dir(
        LOCAL_HARNESS, remote_path="/root/harness", copy=False,
        ignore=[".git", ".git/**", "results/**", "figs/**", "**/__pycache__/**"],
    )
)

app = modal.App("ssd-stale-poc")


@app.function(image=image, volumes={HF_CACHE: hf_vol}, timeout=3600, cpu=4.0, memory=16384)
def download_models():
    """CPU only ($0 GPU). Pre-warms model + dataset cache AND validates every
    dataset loads -- so a dataset bug fails HERE, and run_bench (GPU) never
    spins up for a dataset typo."""
    import sys

    from huggingface_hub import snapshot_download

    for repo in ("Qwen/Qwen3-4B", "deepseek-ai/dspark_qwen3_4b_block7"):
        snapshot_download(repo)
    sys.path.insert(0, "/root/harness")
    import prompts

    prompts.validate_all()   # loads + caches all 3 datasets on CPU
    hf_vol.commit()


@app.function(
    image=image, gpu="A10G",
    volumes={HF_CACHE: hf_vol, OUT_DIR: out_vol},
    timeout=3 * 3600, memory=32768, max_containers=1,
)
def run_bench(gates_only: bool = False):
    import subprocess
    import sys

    # smoke test: sgl-kernel loads on sm_86 AND `import sglang` resolves to the
    # installed editable package (not a shadowing namespace dir) with .Engine.
    subprocess.run(
        [sys.executable, "-c",
         "import sgl_kernel, torch, sglang; "
         "print('sgl_kernel OK', torch.cuda.get_device_name(), '| sglang', sglang.__file__); "
         "assert hasattr(sglang, 'Engine'), 'sglang.Engine missing -- namespace shadowing?'"],
        check=True,
    )
    sys.path.insert(0, "/root/harness")
    from experiment import run_all

    # on_cell_done=out_vol.commit -> results persist to the Volume after EVERY
    # cell, so a hard kill (OOM/timeout) mid-matrix keeps what's done. The finally
    # commit covers ordinary exceptions. A re-run then RESUMES (run_matrix skips
    # any cell whose {key}.json already landed on the Volume) -- no GPU re-paid.
    try:
        run_all(results_dir=OUT_DIR, gates_only=gates_only, on_cell_done=out_vol.commit)
    finally:
        out_vol.commit()


@app.local_entrypoint()
def main(gates_only: bool = False):
    download_models.remote()
    run_bench.remote(gates_only=gates_only)
