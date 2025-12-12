#!/usr/bin/env python3
"""Download Simply models and datasets from HuggingFace.

This script downloads models and datasets to the correct locations
so that config_lib and data_lib can find them automatically.

Usage:
  ./setup_assets.sh                 # Download both models and datasets
  ./setup_assets.sh --models-only   # Download only models
  ./setup_assets.sh --datasets-only # Download only datasets
  ./setup_assets.sh --help          # Show all options

Default locations:
 - Models: ~/.cache/simply/models/
 - Datasets: ~/.cache/simply/datasets/
 - Vocabs: ~/.cache/simply/vocabs/

Override with environment variables:
 - SIMPLY_MODELS
 - SIMPLY_DATASETS
 - SIMPLY_VOCABS
"""

import argparse
import os
import sys
from pathlib import Path

try:
  from huggingface_hub import snapshot_download
except ImportError:
  print("Error: huggingface_hub not found. Install with:")
  print("  pip install huggingface_hub")
  sys.exit(1)


# Default directories (same as config_lib.py and data_lib.py)
MODELS_DIR = os.getenv('SIMPLY_MODELS', os.path.expanduser('~/.cache/simply/models/'))
DATASETS_DIR = os.getenv('SIMPLY_DATASETS', os.path.expanduser('~/.cache/simply/datasets/'))
VOCABS_DIR = os.getenv('SIMPLY_VOCABS', os.path.expanduser('~/.cache/simply/vocabs/'))

# HuggingFace repositories
MODELS_REPO = "unkindledmonkey/simply-models"
DATASETS_REPO = "unkindledmonkey/simply-datasets"


def reorganize_gemma_models(models_dir: str):
  """Reorganize Gemma model directories for Simply's checkpoint format.

  HuggingFace structure:
   GEMMA-2.0-2B-PT-ORBAX/
    ├── tokenizer.model
    └── gemma2-2b/{checkpoint files}

  Required structure (Simply expects step-numbered directories):
   GEMMA-2.0-2B-PT-ORBAX/
    ├── 0/{checkpoint files}
    └── tokenizer.model
  """
  import shutil

  gemma_patterns = [
    ('GEMMA-2.0-2B-PT-ORBAX', 'gemma2-2b'),
    ('GEMMA-2.0-9B-PT-ORBAX', 'gemma2-9b'),
    ('GEMMA-2.0-27B-PT-ORBAX', 'gemma2-27b'),
    ('GEMMA-2.0-2B-IT-ORBAX', 'gemma2-2b-it'),
    ('GEMMA-2.0-9B-IT-ORBAX', 'gemma2-9b-it'),
    ('GEMMA-2.0-27B-IT-ORBAX', 'gemma2-27b-it'),
  ]

  for parent_dir, subdir in gemma_patterns:
    parent_path = Path(models_dir) / parent_dir
    if not parent_path.exists():
      continue

    subdir_path = parent_path / subdir
    step_dir_path = parent_path / '0'  # Simply expects numeric directories

    # Check if reorganization is needed
    if step_dir_path.exists() and (step_dir_path / 'checkpoint').exists():
      continue  # Already reorganized

    print(f"  Reorganizing {parent_dir}...")

    # Create step 0 directory
    step_dir_path.mkdir(exist_ok=True)

    # Move checkpoint files from subdir to 0/ (if subdir exists)
    if subdir_path.exists():
      for item in subdir_path.iterdir():
        if item.name == '.DS_Store':
          continue
        dest = step_dir_path / item.name
        if not dest.exists():
          shutil.move(str(item), str(dest))

      # Remove now-empty subdirectory
      if subdir_path.exists() and not any(subdir_path.iterdir()):
        subdir_path.rmdir()

    # Move checkpoint files from parent level to 0/
    checkpoint_files = ['checkpoint', '_CHECKPOINT_METADATA', '_METADATA',
             'manifest.ocdbt', 'd', 'descriptor', 'ocdbt.process_0']
    for fname in checkpoint_files:
      src = parent_path / fname
      dest = step_dir_path / fname
      if src.exists() and not dest.exists():
        shutil.move(str(src), str(dest))

    # tokenizer.model should stay at parent level for vocab loading
    print("  [OK] Moved checkpoint files to 0/")


def setup_gemma_vocabs(models_dir: str, vocabs_dir: str):
  """Copy Gemma tokenizer files to vocabs directory.

  Simply expects tokenizers in ~/.cache/simply/vocabs/
  """
  import shutil

  Path(vocabs_dir).mkdir(parents=True, exist_ok=True)

  # Map model directories to vocab filenames
  tokenizer_mappings = [
    ('GEMMA-2.0-2B-PT-ORBAX', 'gemma2_tokenizer.model'),
    ('GEMMA-2.0-9B-PT-ORBAX', 'gemma2_tokenizer.model'),
    ('GEMMA-2.0-27B-PT-ORBAX', 'gemma2_tokenizer.model'),
    ('GEMMA-2.0-2B-IT-ORBAX', 'gemma2_tokenizer.model'),
    ('GEMMA-2.0-9B-IT-ORBAX', 'gemma2_tokenizer.model'),
    ('GEMMA-2.0-27B-IT-ORBAX', 'gemma2_tokenizer.model'),
  ]

  for model_dir, vocab_filename in tokenizer_mappings:
    src = Path(models_dir) / model_dir / 'tokenizer.model'
    dest = Path(vocabs_dir) / vocab_filename

    if src.exists() and not dest.exists():
      shutil.copy(str(src), str(dest))
      print(f"  [OK] Copied tokenizer from {model_dir}")


def setup_qwen_vocabs(models_dir: str, vocabs_dir: str):
  """Copy Qwen tokenizer files to vocabs directory.

  Simply expects tokenizers in ~/.cache/simply/vocabs/
  """
  import shutil

  dest_dir = Path(vocabs_dir) / 'Qwen3'
  dest_dir.mkdir(parents=True, exist_ok=True)

  # Source directory in models
  # Assuming the model is downloaded as Qwen3-0.6B
  src_dir = Path(models_dir) / 'Qwen3-0.6B'

  if not src_dir.exists():
    print(f"  [WARNING] Qwen source directory not found: {src_dir}")
    return

  files_to_copy = ['tokenizer.json', 'tokenizer_config.json']
  for filename in files_to_copy:
    src = src_dir / filename
    dest = dest_dir / filename

    if src.exists() and not dest.exists():
      shutil.copy(str(src), str(dest))
      print(f"  [OK] Copied {filename} to {dest_dir}")
    elif not src.exists():
        print(f"  [WARNING] Could not find {filename} in {src_dir}")



def download_models(models_dir: str, repo: str = MODELS_REPO):
  """Download pretrained models from HuggingFace."""
  print(f"Downloading models from {repo}...")
  print(f"Target directory: {models_dir}")

  Path(models_dir).mkdir(parents=True, exist_ok=True)

  try:
    snapshot_download(
      repo_id=repo,
      repo_type="model",
      local_dir=models_dir,
      local_dir_use_symlinks=False,
    )
    print(f"[OK] Models downloaded successfully to {models_dir}")

    # Reorganize Gemma model directories
    print("\nReorganizing Gemma model directories...")
    reorganize_gemma_models(models_dir)

    # Setup vocab files
    print("\nSetting up vocab files...")
    setup_gemma_vocabs(models_dir, VOCABS_DIR)
    setup_qwen_vocabs(models_dir, VOCABS_DIR)

    return True
  except Exception as e:
    print(f"[ERROR] Failed to download models: {e}")
    return False


def download_datasets(datasets_dir: str, repo: str = DATASETS_REPO):
  """Download datasets from HuggingFace."""
  print(f"\nDownloading datasets from {repo}...")
  print(f"Target directory: {datasets_dir}")

  Path(datasets_dir).mkdir(parents=True, exist_ok=True)

  try:
    snapshot_download(
      repo_id=repo,
      repo_type="dataset",
      local_dir=datasets_dir,
      local_dir_use_symlinks=False,
    )
    print(f"[OK] Datasets downloaded successfully to {datasets_dir}")
    return True
  except Exception as e:
    print(f"[ERROR] Failed to download datasets: {e}")
    return False


def check_existing(directory: str) -> bool:
  """Check if directory exists and has content."""
  path = Path(directory)
  if path.exists() and any(path.iterdir()):
    return True
  return False


def main():
  parser = argparse.ArgumentParser(
    description="Download Simply models and datasets from HuggingFace",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=__doc__
  )
  parser.add_argument(
    '--models-only',
    action='store_true',
    help='Download only models'
  )
  parser.add_argument(
    '--datasets-only',
    action='store_true',
    help='Download only datasets'
  )
  parser.add_argument(
    '--models-repo',
    default=MODELS_REPO,
    help=f'HuggingFace models repository (default: {MODELS_REPO})'
  )
  parser.add_argument(
    '--datasets-repo',
    default=DATASETS_REPO,
    help=f'HuggingFace datasets repository (default: {DATASETS_REPO})'
  )
  parser.add_argument(
    '--models-dir',
    default=MODELS_DIR,
    help=f'Models directory (default: {MODELS_DIR})'
  )
  parser.add_argument(
    '--datasets-dir',
    default=DATASETS_DIR,
    help=f'Datasets directory (default: {DATASETS_DIR})'
  )
  parser.add_argument(
    '--force',
    action='store_true',
    help='Download even if directories already exist'
  )

  args = parser.parse_args()

  # Determine what to download
  download_models_flag = not args.datasets_only
  download_datasets_flag = not args.models_only

  print("=" * 70)
  print("Simply Assets Download Script")
  print("=" * 70)

  success = True

  # Download models
  if download_models_flag:
    if not args.force and check_existing(args.models_dir):
      print(f"\n[WARNING] Models directory already exists: {args.models_dir}")
      print("   Use --force to re-download")
    else:
      if not download_models(args.models_dir, args.models_repo):
        success = False

  # Download datasets
  if download_datasets_flag:
    if not args.force and check_existing(args.datasets_dir):
      print(f"\n[WARNING] Datasets directory already exists: {args.datasets_dir}")
      print("   Use --force to re-download")
    else:
      if not download_datasets(args.datasets_dir, args.datasets_repo):
        success = False

  print("\n" + "=" * 70)
  if success:
    print("[SUCCESS] Download complete!")
    print("\nDirectory structure:")
    print(f"  Models:   {args.models_dir}")
    print(f"  Datasets: {args.datasets_dir}")
    print(f"  Vocabs:   {VOCABS_DIR} (created on first use)")
    print("\nYou can now run experiments with:")
    print("  python -m simply.main --experiment_config <config_name> ...")
  else:
    print("[FAILED] Download failed. See errors above.")
    sys.exit(1)


if __name__ == '__main__':
  main()
