import os
import subprocess
from ftplib import FTP

# FTP configuration
FTP_HOST = "lab.rfmri.org"
FTP_USER = "ftpdownload"
FTP_PASS = "FTPDownload"
BASE_PATH = "/sharing/RfMRIMaps"

# Define your datasets and their file paths (relative to BASE_PATH)
DATASETS = {
    "ABIDE": [
        "ABIDE/ABIDE_RfMRI.part1.rar",
        "ABIDE/ABIDE_RfMRI.part2.rar",
        "ABIDE/ABIDE_RfMRI.part3.rar",
        "ABIDE/RfMRIMaps_ABIDE_Phenotypic.csv",
    ],
    "ABIDE2": [
        "ABIDE2/ABIDE2_RfMRI.part1.rar",
        "ABIDE2/ABIDE2_RfMRI.part2.rar",
        "ABIDE2/ABIDE2_RfMRI.part3.rar",
        "ABIDE2/RfMRIMaps_ABIDE2_Phenotypic.csv",
    ],
    "ADHD": [
        "ADHD200/ADHD200_RfMRI.part1.rar",
        "ADHD200/ADHD200_RfMRI.part2.rar",
        "ADHD200/ADHD200_RfMRI.part3.rar",
        "ADHD200/ADHD200_Phenotypic.csv",
    ],
    "Beijing_EOEC": [
        "Beijing_EOEC/Beijing_EOEC1.tar.gz",
        "Beijing_EOEC/Beijing_EOEC2.tar.gz",
        "Beijing_EOEC/Beijing_EOEC1_phenographic.csv",
        "Beijing_EOEC/Beijing_EOEC2_phenographic.csv",
    ],
    "CORR": [
        "CORR/CORR_BetweenSessionTRT/CORR_BetweenSessionTRT_RfMRI.part1.rar",
        "CORR/CORR_BetweenSessionTRT/CORR_BetweenSessionTRT_RfMRI.part2.rar",
        "CORR/CORR_BetweenSessionTRT/CORR_BetweenSessionTRT_RfMRI.part3.rar",
        "CORR/CORR_HNU_CCBD_10Repetition/CORR_HNU_CCBD10Repetition.rar",
        "CORR/CORR_WithinSessionTRT/CORR_WithinSessionTRT.part1.rar",
        "CORR/CORR_WithinSessionTRT/CORR_WithinSessionTRT.part2.rar",
        "CORR/SubjectID_BetweenSessionTRT.csv",
        "CORR/SubjectID_WithinSessionTRT.csv",
        "CORR/SubjectID_HNU_CCBD_10Repetitions.csv",
    ],
    "FCP": [],  # Automatically fetch all files from the folder
    "Rumination_fMRI": [
        "PaperDataSharing/Chen_2020_RuminationfMRIData/RuminationfMRIData.tar.gz",
        "PaperDataSharing/Chen_2020_RuminationfMRIData/RuminationfMRIData_Phenotypic.csv",
    ]
}

def list_files_in_ftp_dir(ftp: FTP, path: str):
    ftp.cwd(path)
    return ftp.nlst()

def validate_and_expand_files():
    ftp = FTP(FTP_HOST)
    ftp.login(FTP_USER, FTP_PASS)
    validated = {}

    for dataset, file_list in DATASETS.items():
        if not file_list:
            # Auto-fetch all files from folder if list is empty
            folder = f"{BASE_PATH}/{dataset}"
            try:
                print(f"🔍 Auto-listing files for dataset: {dataset}")
                ftp.cwd(folder)
                file_list = ftp.nlst()
                file_list = [f"{dataset}/{f}" for f in file_list]
            except Exception as e:
                raise RuntimeError(f"❌ Failed to list files in {folder}: {e}")

        missing_files = []
        verified_files = []

        for path in file_list:
            folder, filename = os.path.split(path)
            try:
                ftp.cwd(f"{BASE_PATH}/{folder}")
                if filename not in ftp.nlst():
                    missing_files.append(path)
                else:
                    verified_files.append(path)
            except Exception as e:
                raise RuntimeError(f"❌ FTP error checking file {path}: {e}")

        if missing_files:
            raise FileNotFoundError(f"❌ Missing files in dataset '{dataset}': {missing_files}")

        validated[dataset] = verified_files

    ftp.quit()
    return validated

def download_with_aria2(ftp_path: str, output_dir: str):
    ftp_url = f"ftp://{FTP_USER}:{FTP_PASS}@{FTP_HOST}{BASE_PATH}/{ftp_path}"
    output_file = os.path.basename(ftp_path)

    command = [
        "aria2c",
        "--ftp-user=" + FTP_USER,
        "--ftp-passwd=" + FTP_PASS,
        "--split=16",
        "--max-connection-per-server=16",
        "--continue=true",
        "-o", output_file,
        ftp_url
    ]

    print(f"⬇️ Downloading {output_file} ...")
    subprocess.run(command, check=True)
    print(f"✅ Downloaded: {output_file}")

def main(output_root="/datastorage/li/fMRI"):
    os.makedirs(output_root, exist_ok=True)
    print("🔍 Validating files before downloading...")
    validated_datasets = validate_and_expand_files()

    for dataset, files in validated_datasets.items():
        print(f"\n📦 Dataset: {dataset}")
        dataset_dir = os.path.join(output_root, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        os.chdir(dataset_dir)

        for file_path in files:
            try:
                download_with_aria2(file_path, dataset_dir)
            except Exception as e:
                print(f"⚠️ Failed to download {file_path}: {e}")

if __name__ == "__main__":
    main()
