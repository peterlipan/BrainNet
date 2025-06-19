import os
import re
import shutil
import tarfile
import torch
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

def parse_csv_to_tensor_list(df: pd.DataFrame):
    """
    Parses a DataFrame with fMRI columns named like [preprocess]_s[s]_g[g]_[t]
    and returns a list of dicts with metadata and tensors.
    """
    groups = defaultdict(list)
    pattern = re.compile(
        r'^(?P<preprocess>[a-zA-Z0-9]+)_s(?P<scrub>[01])_g(?P<gsr>[01])_(?P<time>\d+)$'
    )

    for col in df.columns:
        match = pattern.match(col)
        if match:
            key = (
                match.group("preprocess"),
                match.group("scrub"),
                match.group("gsr")
            )
            time_idx = int(match.group("time"))
            groups[key].append((time_idx, col))

    result = []
    for (preprocess, scrub, gsr), items in groups.items():
        sorted_cols = [col for _, col in sorted(items, key=lambda x: x[0])]
        data_tensor = torch.tensor(df[sorted_cols].values, dtype=torch.float32)
        result.append({
            "preprocess": preprocess,
            "scrubbing": scrub,
            "global_signal_regression": gsr,
            "data": data_tensor
        })

    return result


def extract_tar_gz(file_path, extract_to):
    """Safely extract a tar.gz file to a given directory."""
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_to)


def organize_ACPI(src: str, dst: str):
    print("🚀 Starting ACPI dataset organization...")

    src_path = os.path.join(src, 'ACPI')
    dst_path = os.path.join(dst, 'ACPI')
    temp_path = os.path.join(dst_path, 'temp')
    os.makedirs(temp_path, exist_ok=True)

    file_patterns = [
        "mta_1_ts_{}_rois.tar.gz",
        "nyu_1_ts_{}_rois.tar.gz"
    ]
    rois = ['aal', 'cc200', 'ho', 'rand3200']
    files = [os.path.join(src_path, pat.format(roi)) for pat in file_patterns for roi in rois]

    print("📦 Verifying input files...")
    for file in files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"❌ Missing expected file: {file}")

    print("🗃️ Extracting archives...")
    for file in tqdm(files, desc="📤 Extracting"):
        dataset = os.path.basename(file).split('_')[0]
        roi = os.path.basename(file).split('_')[3]
        output_dir = os.path.join(temp_path, dataset, roi)
        os.makedirs(output_dir, exist_ok=True)
        extract_tar_gz(file, output_dir)

    print("🔍 Organizing extracted files...")
    records = []
    atlases = ';'.join(rois)
    for dataset in tqdm(['mta', 'nyu'], desc="📁 Datasets"):
        for roi in tqdm(rois, desc=f"📚 ROIs for {dataset}", leave=False):
            save_path = os.path.join(dst_path, roi)
            os.makedirs(save_path, exist_ok=True)

            path = os.path.join(temp_path, dataset, roi)
            folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

            for folder in tqdm(folders, desc=f"🧠 {roi}/{dataset}", leave=False):
                folder_path = os.path.join(path, folder)

                try:
                    subject_id, session_tag = folder.split('-', 1)
                    session_id = session_tag.split('_')[1]
                except Exception:
                    print(f"⚠️ Skipping malformed folder name: {folder}")
                    continue

                sid = f"{dataset}_{subject_id}"
                data_path = os.path.join(folder_path, f'ts_{roi}_rois.csv')
                if not os.path.exists(data_path):
                    print(f"⚠️ Missing CSV: {data_path}")
                    continue

                try:
                    df = pd.read_csv(data_path)
                except Exception as e:
                    print(f"❌ Error reading {data_path}: {e}")
                    continue

                samples = parse_csv_to_tensor_list(df)
                for sample in samples:
                    preprocess = sample['preprocess']
                    scrub = sample['scrubbing']
                    gsr = sample['global_signal_regression']
                    data = sample['data']
                    filename = f"{sid}_{session_id}_{preprocess}_s{scrub}_g{gsr}.pt"
                    torch.save(data, os.path.join(save_path, filename))
                    records.append({
                        'Subject.ID': sid,
                        'File.Session': session_id,
                        'File.Preprocess': preprocess,
                        'File.Scrubbing': scrub,
                        'File.Global_signal_regression': gsr,
                        'File.Atlases': atlases,
                        'File.Name': filename,
                    })

    print("💾 Saving phenotype file...")
    records_df = pd.DataFrame.from_records(records)
    records_df.to_csv(os.path.join(dst_path, 'ACPI_Phenotypes.csv'), index=False)

    print("🧹 Cleaning up temporary files...")
    shutil.rmtree(temp_path)
    shutil.rmtree(src_path)

    print("✅ Done! Processed data saved to:", dst_path)


if __name__ == "__main__":
    src = '/datastorage/li/fMRI'
    dst = '/datastorage/li/BrainNeDatasets'
    
    organize_ACPI(src, dst)
    