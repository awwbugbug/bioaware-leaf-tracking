from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="shl-shawn/CanolaTrack",
    repo_type="dataset",
    local_dir=r"D:\VScode_file_2\LeafTrackNet-main\datasets\CanolaTrack",
    local_dir_use_symlinks=False,
    max_workers=2,  # 降低并发，避免 429
    token=True,  # 使用你 hf auth login 后的本地 token
)

print("Download finished.")
