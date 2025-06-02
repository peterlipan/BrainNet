import os
import yaml
import subprocess
from ftplib import FTP


# FTP configuration
FTP_HOST = "lab.rfmri.org"
FTP_USER = "ftpdownload"
FTP_PASS = "FTPDownload"


def load_datasets_from_yaml(path="datasets.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ftp_file_exists(ftp, full_path):
    """Check if a file exists on the FTP server by navigating its full path."""
    folder, filename = os.path.split(full_path)
    try:
        original_dir = ftp.pwd()
        for subdir in folder.strip("/").split("/"):
            ftp.cwd(subdir)
        file_list = ftp.nlst()
        ftp.cwd(original_dir)  # Restore
        return filename in file_list
    except Exception as e:
        raise RuntimeError(f"❌ FTP error checking file {full_path}: {e}")


def validate_and_expand_files(datasets):
    ftp = FTP(FTP_HOST)
    ftp.login(FTP_USER, FTP_PASS)
    validated = {}

    for dataset, file_list in datasets.items():
        missing_files = []
        verified_files = []

        if not file_list:
            print(f"⚠️ Dataset '{dataset}' has no files listed.")
            continue

        for path in file_list:
            if ftp_file_exists(ftp, path):
                verified_files.append(path)
            else:
                missing_files.append(path)

        if missing_files:
            raise FileNotFoundError(f"❌ Missing files in dataset '{dataset}': {missing_files}")
        validated[dataset] = verified_files

    ftp.quit()
    return validated


def download_with_aria2(ftp_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.basename(ftp_path)
    output_path = os.path.join(output_dir, output_file)

    # ✅ Skip download if file already exists
    if os.path.exists(output_path):
        print(f"⏭️ Skipping {output_file}, already exists in {output_dir}")
        return

    ftp_url = f"ftp://{FTP_USER}:{FTP_PASS}@{FTP_HOST}{ftp_path}"

    command = [
        "aria2c",
        "--ftp-user=" + FTP_USER,
        "--ftp-passwd=" + FTP_PASS,
        "--split=16",
        "--max-connection-per-server=16",
        "--continue=true",
        "-d", output_dir,
        "-o", output_file,
        ftp_url
    ]

    print(f"⬇️ Downloading {output_file} to {output_dir} ...")
    subprocess.run(command, check=True)
    print(f"✅ Downloaded: {output_file}")



def main(output_root="/datastorage/li/fMRI", dataset_yaml="RawDatasets.yaml"):
    os.makedirs(output_root, exist_ok=True)
    datasets = load_datasets_from_yaml(dataset_yaml)

    print("🔍 Validating files on FTP...")
    validated_datasets = validate_and_expand_files(datasets)

    for dataset, files in validated_datasets.items():
        print(f"\n📦 Dataset: {dataset}")
        dataset_dir = os.path.join(output_root, dataset)
        for file_path in files:
            try:
                download_with_aria2(file_path, dataset_dir)
            except Exception as e:
                print(f"⚠️ Failed to download {file_path}: {e}")


if __name__ == "__main__":
    main()
