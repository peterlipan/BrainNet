import os
import subprocess
import re
import argparse

def get_subject_ids(s3_base):
    """
    Retrieves a list of subject IDs from the S3 bucket.
    """
    try:
        result = subprocess.run(
            ["aws", "s3", "ls", s3_base, "--no-sign-request"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        subject_ids = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                folder = parts[-1].strip('/')
                if "sub-" in folder:  # Check if the folder name contains 'sub-'
                    print(f"Found subject ID: {folder}")
                    subject_ids.append(folder)
        return subject_ids
    except subprocess.CalledProcessError as e:
        print(f"Error listing S3 directories: {e.stderr}")
        return []

def download_roi_timeseries(subject_id, s3_base, output_dir):
    """
    Downloads the roi_timeseries or roi_timeseries_0 folder for a given subject ID.
    """
    possible_folders = ["roi_timeseries", "roi_timeseries_0"]
    folder_to_download = None

    for folder_name in possible_folders:
        s3_check_path = f"{s3_base}{subject_id}/{folder_name}/"
        try:
            result = subprocess.run(
                ["aws", "s3", "ls", s3_check_path, "--no-sign-request"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            if result.stdout.strip():
                folder_to_download = folder_name
                break
        except subprocess.CalledProcessError:
            continue  # Try next folder

    if folder_to_download is None:
        print(f"❌ No valid roi_timeseries folder found for {subject_id}")
        return

    s3_path = f"{s3_base}{subject_id}/{folder_to_download}/"
    local_dir = os.path.join(output_dir, subject_id)
    os.makedirs(local_dir, exist_ok=True)

    print(f"⬇️ Downloading {folder_to_download} for {subject_id} ...")

    cmd = [
        "aws", "s3", "cp",
        s3_path,
        local_dir,
        "--recursive",
        "--no-sign-request"
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Done: {subject_id}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {subject_id} — {e}")


def main():
    parser = argparse.ArgumentParser(description="Download roi_timeseries from HBN S3 bucket.")
    parser.add_argument("--s3-base", type=str, required=True, help="S3 base path, e.g. s3://fcp-indi/data/Projects/HBN/CPAC_preprocessed_Derivatives/")
    parser.add_argument("--output-dir", type=str, required=True, help="Local output directory")

    args = parser.parse_args()
    s3_base = args.s3_base.rstrip('/') + '/'
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    subject_ids = get_subject_ids(s3_base)
    if not subject_ids:
        print("No subject IDs found.")
        return
    for subject_id in subject_ids:
        download_roi_timeseries(subject_id, s3_base, output_dir)

if __name__ == "__main__":
    main()
