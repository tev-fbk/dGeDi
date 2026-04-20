from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="ahamza848/dGeDi",
    filename="dGeDi_multi_scale.pth",
    local_dir="checkpoints"
    )

hf_hub_download(
    repo_id="ahamza848/dGeDi",
    filename="dGeDi_single_scale.pth",
    local_dir="checkpoints"
    )