"""
data/download_wlasl.py
----------------------
Downloads the WLASL (Word-Level American Sign Language) dataset.

WLASL contains ~21,000 video clips across 2000 ASL words.
We download only the top-100 subset defined in word_list.json.

Usage:
    python data/download_wlasl.py [--output-dir data/raw] [--top-n 100]

Dataset source: https://dxli94.github.io/WLASL/
Backup mirror:  Kaggle — search "WLASL dataset"
"""

import argparse
import json
import os
import sys
from pathlib import Path

import gdown
import requests
from tqdm import tqdm


# ─── Official WLASL metadata JSON ─────────────────────────────────────────────
WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"

# Google Drive IDs for WLASL video archives (official mirrors)
# If these go stale, use the Kaggle backup (see README).
WLASL_GDRIVE_IDS = {
    "WLASL300":  "1qbKNEBEDOoC-kRB8j5lTROqNEiKDrEnP",
    "WLASL1000": "1JkKn8SXJE3cM5U_t5KDFVqxiFXF6Y97U",
    "WLASL2000": "1uMpBMdRzqJHEGKYn7GRCiSE6EY3_2hkO",
}


def load_word_list(word_list_path: str) -> list[str]:
    """Load the target word list from word_list.json."""
    with open(word_list_path) as f:
        data = json.load(f)
    return [w.upper() for w in data["words"]]  # WLASL uses uppercase glosses


def download_wlasl_metadata(output_dir: Path) -> Path:
    """Download the WLASL metadata JSON (video URLs + labels)."""
    meta_path = output_dir / "WLASL_v0.3.json"
    if meta_path.exists():
        print(f"[skip] Metadata already exists: {meta_path}")
        return meta_path

    print("Downloading WLASL metadata JSON...")
    response = requests.get(WLASL_JSON_URL, timeout=30)
    response.raise_for_status()
    with open(meta_path, "w") as f:
        f.write(response.text)
    print(f"[ok] Saved metadata to {meta_path}")
    return meta_path


def filter_metadata(meta_path: Path, target_words: list[str]) -> dict:
    """
    Parse WLASL metadata and return only entries for our top-100 words.

    Returns:
        {
          "HELLO": [{"video_id": "...", "url": "...", "split": "train"}, ...],
          ...
        }
    """
    with open(meta_path) as f:
        wlasl_data = json.load(f)

    target_set = set(target_words)
    filtered = {}
    total_clips = 0

    for entry in wlasl_data:
        gloss = entry["gloss"].upper()
        if gloss not in target_set:
            continue

        clips = []
        for instance in entry.get("instances", []):
            clips.append({
                "video_id": instance["video_id"],
                "url":      instance.get("url", ""),
                "split":    instance.get("split", "train"),
                "signer":   instance.get("signer_id", -1),
                "start":    instance.get("frame_start", -1),
                "end":      instance.get("frame_end", -1),
            })
        if clips:
            filtered[gloss] = clips
            total_clips += len(clips)

    print(f"Found {len(filtered)} words / {total_clips} clips in metadata")
    return filtered


def download_videos(filtered: dict, output_dir: Path, max_per_class: int = 100):
    """
    Download video clips for each word.
    Falls back to skipping if a URL is dead (common in WLASL).
    """
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    downloaded = 0

    for gloss, clips in filtered.items():
        word_dir = videos_dir / gloss
        word_dir.mkdir(exist_ok=True)

        clips_to_get = clips[:max_per_class]
        for clip in tqdm(clips_to_get, desc=gloss, leave=False):
            vid_id = clip["video_id"]
            out_path = word_dir / f"{vid_id}.mp4"

            if out_path.exists():
                continue  # already downloaded

            url = clip.get("url", "")
            if not url:
                skipped += 1
                continue

            try:
                r = requests.get(url, timeout=15, stream=True)
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                downloaded += 1
            except Exception as e:
                skipped += 1

    print(f"\n[done] Downloaded: {downloaded} | Skipped/Failed: {skipped}")
    print(f"Videos saved to: {videos_dir}")


def download_from_gdrive(output_dir: Path, archive_key: str = "WLASL300"):
    """
    Alternative: download pre-packaged archive from Google Drive.
    Useful if individual video URLs are mostly dead.
    """
    gdrive_id = WLASL_GDRIVE_IDS.get(archive_key)
    if not gdrive_id:
        print(f"Unknown archive key: {archive_key}. Choose from {list(WLASL_GDRIVE_IDS)}")
        sys.exit(1)

    out_path = output_dir / f"{archive_key}.zip"
    if out_path.exists():
        print(f"[skip] Archive already exists: {out_path}")
        return

    print(f"Downloading {archive_key} from Google Drive (~several GB, please wait)...")
    gdown.download(id=gdrive_id, output=str(out_path), quiet=False)
    print(f"[ok] Saved to {out_path}")
    print(f"Extract with: unzip {out_path} -d {output_dir}/videos/")


def save_split_manifest(filtered: dict, output_dir: Path):
    """
    Save train/val/test split info as a JSON manifest.
    Used by data/dataset.py to build the DataLoader.
    """
    manifest = {"train": [], "val": [], "test": []}
    word_to_idx = {word: i for i, word in enumerate(sorted(filtered.keys()))}

    for gloss, clips in filtered.items():
        label = word_to_idx[gloss]
        for clip in clips:
            split = clip["split"]
            if split not in manifest:
                split = "train"  # default unknown splits to train
            manifest[split].append({
                "video_id": clip["video_id"],
                "gloss":    gloss,
                "label":    label,
                "npy_path": f"data/keypoints/{gloss}/{clip['video_id']}.npy",
            })

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"word_to_idx": word_to_idx, "splits": manifest}, f, indent=2)

    print(f"[ok] Manifest saved to {manifest_path}")
    for split, items in manifest.items():
        print(f"     {split}: {len(items)} clips")


def main():
    parser = argparse.ArgumentParser(description="Download WLASL dataset")
    parser.add_argument("--output-dir",    default="data/raw",  help="Where to save raw data")
    parser.add_argument("--word-list",     default="data/word_list.json")
    parser.add_argument("--gdrive",        action="store_true", help="Use Google Drive archive instead of individual URLs")
    parser.add_argument("--gdrive-key",    default="WLASL300",  help="Which GDrive archive: WLASL300 | WLASL1000 | WLASL2000")
    parser.add_argument("--max-per-class", type=int, default=100, help="Max clips to download per word")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  WLASL Dataset Downloader")
    print("=" * 60)

    # Step 1: load target word list
    target_words = load_word_list(args.word_list)
    print(f"Target words: {len(target_words)}")

    # Step 2: download metadata JSON
    meta_path = download_wlasl_metadata(output_dir)

    # Step 3: filter to our 100 words
    filtered = filter_metadata(meta_path, target_words)

    # Step 4: save manifest (always, even before downloading videos)
    save_split_manifest(filtered, output_dir)

    # Step 5: download videos
    if args.gdrive:
        download_from_gdrive(output_dir, args.gdrive_key)
    else:
        print("\nDownloading individual video clips (this may take a while)...")
        print("TIP: If many URLs fail, re-run with --gdrive flag\n")
        download_videos(filtered, output_dir, max_per_class=args.max_per_class)


if __name__ == "__main__":
    main()
