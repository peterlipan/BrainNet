import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset


class RfMRIDataset(Dataset):
    def __init__(self, data_path, dataset_name, transform=None, seed=None):
        self.data_path = data_path
        self.transform = transform
        self.dataset_name = dataset_name
        self.phenotype_filename = f"{dataset_name}_Phenotypes.csv"
        self.phenotype_csv = pd.read_csv(os.path.join(data_path, self.phenotype_filename))

        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random

        self.subject_ids = self.phenotype_csv['Subject.ID'].unique()
        self.n_subjects = len(self.subject_ids)
        # consistent atlases across all subjects
        self.atlas_list = self.phenotype_csv['File.Atlases'].values[0].split(';')
        self.n_atlases = len(self.atlas_list)
        self.n_files = len(self.phenotype_csv) * self.n_atlases

    def _sample_from_subject(self, subject_id):
        """
        Given a subject ID, randomly return paths of two files from the subject's data.
        """
        rows = self.phenotype_csv[self.phenotype_csv['Subject.ID'] == subject_id]
        # there should be two preprocessing methods for RfMRI datasets,
        # which can be regarded as temporally matched
        filenames = rows['File.Name'].values
        path_list = [os.path.join(self.data_path, atlas, filename) for atlas in self.atlas_list for filename in filenames]
        path_a, path_b = self.rng.sample(path_list, 2)
        return path_a, path_b

    def __len__(self):
        # return the number of subjects
        return self.n_subjects

    def __getitem__(self, idx):
        # sample on the subjects
        sid = self.subject_ids[idx]
        path_a, path_b = self._sample_from_subject(sid)
        # load the data
        data_a = torch.load(path_a)
        data_b = torch.load(path_b)
        # apply transformations if any
        if self.transform:
            data_a = self.transform(data_a)
            data_b = self.transform(data_b)
        return data_a, data_b


class ABIDEDataset(RfMRIDataset):
    def __init__(self, data_path, transform=None, seed=None):
        super().__init__(data_path, 'ABIDE', transform, seed)

class ABIDE2Dataset(RfMRIDataset):
    def __init__(self, data_path, transform=None, seed=None):
        super().__init__(data_path, 'ABIDE2', transform, seed)

class ADHDDataset(RfMRIDataset):
    def __init__(self, data_path, transform=None, seed=None):
        super().__init__(data_path, 'ADHD', transform, seed)

class BeijingEOECDataset(RfMRIDataset):
    def __init__(self, data_path, transform=None, seed=None):
        super().__init__(data_path, 'Beijing_EOEC', transform, seed)

class CORRDataset(RfMRIDataset):
    def __init__(self, data_path, transform=None, seed=None):
        super().__init__(data_path, 'CORR', transform, seed)

class FCPDataset(RfMRIDataset):
    def __init__(self, data_path, transform=None, seed=None):
        super().__init__(data_path, 'FCP', transform, seed)

class RestMetaMddDataset(RfMRIDataset):
    def __init__(self, data_path, transform=None, seed=None):
        super().__init__(data_path, 'REST-meta-MDD', transform, seed)
