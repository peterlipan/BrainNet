import os
import shutil
import torch
import rarfile
import scipy.io as sio
import numpy as np
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod


def segment_feature_by_atlas(features: np.ndarray) -> dict:
    if features.ndim != 2:
        raise ValueError("Input must be a 2D array of shape [L, N]")

    _, N = features.shape
    atlas_splits = [
        (0, 116, "AAL"),
        (116, 228, "HO"),
        (228, 428, "CC200"),
        (428, 1408, "ZALE"),
        (1408, 1568, "DOS160"),
        (1568, 1569, "GLOBAL"),
        (1569, 1833, "PWR264"),
        (1833, 2233, "SCH400"),
    ]
    valid_lengths = [end for _, end, _ in atlas_splits]

    if N not in valid_lengths:
        raise ValueError(f"Invalid feature width N={N}. Must be one of {valid_lengths}")

    output = {}
    for start, end, name in atlas_splits:
        if N >= end:
            output[name] = features[:, start:end]
        else:
            break

    output.pop("GLOBAL", None)
    return output


class DatasetOrganizer(ABC):
    def __init__(self, src, dst, dataset_name):
        self.src_path = os.path.join(src, dataset_name)
        self.dst_path = os.path.join(dst, dataset_name)
        self.temp_path = os.path.join(self.dst_path, 'temp')
        os.makedirs(self.temp_path, exist_ok=True)
        self.records = []
        self.roi_folders = ['ROISignals_FunImgARCWF', 'ROISignals_FunImgARglobalCWF']

    @abstractmethod
    def get_required_files(self):
        pass

    @abstractmethod
    def load_phenotypic_data(self):
        pass

    @abstractmethod
    def parse_subject_row(self, row):
        pass

    def extract_archive(self, archive_path):
        print(f"📦 Extracting {os.path.basename(archive_path)}...")
        with rarfile.RarFile(archive_path) as rf:
            rf.extractall(self.temp_path)
        print(f"✅ Extracted to: {self.temp_path}")

    def validate_files(self):
        for file in self.get_required_files():
            if not os.path.exists(file):
                raise FileNotFoundError(f"❌ File not found: {file}")
        print(f"✅ All files are present for {self.__class__.__name__}.")

    def organize(self):
        self.validate_files()
        self.extract_archive(self.get_required_files()[0])
        phenotypic_data = self.load_phenotypic_data()
        print(f"📄 Loaded phenotypic data ({len(phenotypic_data)} samples)")

        results_path = os.path.join(self.temp_path, 'Results')

        for _, row in tqdm(phenotypic_data.iterrows(), total=len(phenotypic_data), desc="🧠 Processing subjects"):
            subject_info = self.parse_subject_row(row)
            for roi_folder in self.roi_folders:
                file_path = os.path.join(results_path, roi_folder, f'ROISignals_{subject_info["id"]}.mat')
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"❌ Required file missing: {file_path}")
                data = sio.loadmat(file_path)['ROISignals']
                atlas_data = segment_feature_by_atlas(data)
                filename = f'{roi_folder}_{subject_info["id"]}.pt'
                subject_info['atlases'] = ";".join(atlas_data.keys())
                subject_info['filename'] = filename
                self.records.append(subject_info)

                for atlas_name, atlas_features in atlas_data.items():
                    save_path = os.path.join(self.dst_path, atlas_name, filename)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save(torch.tensor(atlas_features, dtype=torch.float32), save_path)

        self.finalize()

    def finalize(self):
        df = pd.DataFrame.from_records(self.records)
        df.to_csv(os.path.join(self.dst_path, f'{os.path.basename(self.dst_path)}_Phenotypes.csv'), index=False)
        print(f"💾 Saved phenotype table with {len(df)} entries.")

        shutil.rmtree(self.src_path)
        shutil.rmtree(self.temp_path)
        print(f"🧹 Cleaned up temporary and source directories.")


class ABIDEDataset(DatasetOrganizer):
    def __init__(self, src, dst):
        super().__init__(src, dst, 'ABIDE')

    def get_required_files(self):
        return [
            os.path.join(self.src_path, f'ABIDE_RfMRI.part{i}.rar') for i in range(1, 4)
        ] + [os.path.join(self.src_path, 'RfMRIMaps_ABIDE_Phenotypic.csv')]

    def load_phenotypic_data(self):
        return pd.read_csv(self.get_required_files()[-1])

    def parse_subject_row(self, row):
        return {
            'id': f'{int(row["SUB_ID"]):07d}',
            'diagnosis': 2 - row['DX_GROUP'],
            'subtype': row['DSM_IV_TR'],
            'age': row['AGE_AT_SCAN'],
            'gender': 2 - row['SEX'],
        }


class ABIDE2Dataset(DatasetOrganizer):
    def __init__(self, src, dst):
        super().__init__(src, dst, 'ABIDE2')

    def get_required_files(self):
        return [
            os.path.join(self.src_path, f'ABIDE2_RfMRI.part{i}.rar') for i in range(1, 4)
        ] + [os.path.join(self.src_path, 'RfMRIMaps_ABIDE2_Phenotypic.csv')]

    def load_phenotypic_data(self):
        return pd.read_csv(self.get_required_files()[-1], encoding='latin1')

    def parse_subject_row(self, row):
        return {
            'id': str(row['SUB_LIST']),
            'diagnosis': 2 - row['DX_GROUP'],
            'subtype': row['PDD_DSM_IV_TR'],
            'age': row['AGE_AT_SCAN '],
            'gender': 2 - row['SEX'],
        }


class ADHDDataset(DatasetOrganizer):
    def __init__(self, src, dst):
        super().__init__(src, dst, 'ADHD')

    def get_required_files(self):
        return [
            os.path.join(self.src_path, f'ADHD200_RfMRI.part{i}.rar') for i in range(1, 4)
        ] + [os.path.join(self.src_path, 'ADHD200_Phenotypic.csv')]

    def load_phenotypic_data(self):
        return pd.read_csv(self.get_required_files()[-1])

    def parse_subject_row(self, row):
        subtype = row['DX'].replace('\'', '')
        if subtype == 'pending':
            return None  # skip
        subtype = int(subtype)
        return {
            'id': row['Participant ID'].replace('\'', ''),
            'diagnosis': 1 if subtype > 0 else 0,
            'subtype': subtype,
            'age': row['Age'],
            'gender': row['Gender'],
        }


class RESTMDDDataset(DatasetOrganizer):
    def __init__(self, src, dst):
        super().__init__(src, dst, 'REST-meta-MDD')

    def get_required_files(self):
        zip_file = os.path.join(self.src_path, 'REST-meta-MDD-Phase1-Sharing.zip')
        folder = os.path.join(self.src_path, 'REST-meta-MDD-Phase1-Sharing')
        excel = os.path.join(folder, 'REST-meta-MDD-PhenotypicData_WithHAMDSubItem_V4.xlsx')
        assert os.path.exists(zip_file)
        assert os.path.exists(folder)
        assert os.path.exists(excel)
        return [zip_file, excel]

    def extract_archive(self, archive_path):
        print("📝 Please manually extract the REST-meta-MDD zip and provide password. Skipping auto extraction.")

    def load_phenotypic_data(self):
        excel_path = self.get_required_files()[1]
        mdd_df = pd.read_excel(excel_path, sheet_name="MDD", usecols=["ID", "Sex", "Age"])
        ctrl_df = pd.read_excel(excel_path, sheet_name="Controls", usecols=["ID", "Sex", "Age"])
        mdd_df["Diagnosis"] = 1
        ctrl_df["Diagnosis"] = 0
        return pd.concat([mdd_df, ctrl_df], ignore_index=True)

    def parse_subject_row(self, row):
        return {
            'id': row['ID'],
            'diagnosis': row['Diagnosis'],
            'subtype': row['Diagnosis'],
            'age': row['Age'],
            'gender': row['Sex'],
        }


class BeijingEOECDataset(DatasetOrganizer):
    def __init__(self, src, dst):
        super().__init__(src, dst, 'Beijing_EOEC')

    def get_required_files(self):
        return [
            os.path.join(self.src_path, 'Beijing_EOEC1.tar.gz'),
            os.path.join(self.src_path, 'Beijing_EOEC2.tar.gz'),
            os.path.join(self.src_path, 'Beijing_EOEC1_phenographic.csv'),
            os.path.join(self.src_path, 'Beijing_EOEC2_phenographic.csv'),
        ]

    def extract_archive(self, archive_path):
        print(f"📦 Extracting {os.path.basename(archive_path)}...")
        os.system(f'tar -xzf {archive_path} -C {self.temp_path}')
        print(f"✅ Extracted {os.path.basename(archive_path)}")

    def load_phenotypic_data(self):
        dfs = [
            pd.read_csv(self.get_required_files()[2]),
            pd.read_csv(self.get_required_files()[3])
        ]
        return pd.concat(dfs, ignore_index=True)

    def parse_subject_row(self, row):
        return {
            'id': str(row['Subject_ID']),
            'diagnosis': int(row['Group']),  # Ensure Group is 0 or 1
            'subtype': int(row['Group']),
            'age': float(row['Age']),
            'gender': int(row['Gender']),
        }



if __name__ == "__main__":
    src = '/datastorage/li/fMRI'
    dst = '/datastorage/li/BrainNeDatasets'

    # ABIDEDataset(src, dst).organize()
    # ABIDE2Dataset(src, dst).organize()
    # ADHDDataset(src, dst).organize()
    RESTMDDDataset(src, dst).organize()
