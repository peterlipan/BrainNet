from torch.utils.data import Dataset


class PretrainDataset(Dataset):
    def __init__(self, dataset_list):
        """
        dataset_list: list of RfMRIDataset instances
        """
        self.datasets = dataset_list
        self.subject_index_map = []  # list of (dataset_idx, subject_idx)
        
        # Build a mapping from global index to (dataset index, subject index)
        for d_idx, dataset in enumerate(self.datasets):
            for s_idx in range(len(dataset)):
                self.subject_index_map.append((d_idx, s_idx))
        
        self.n_subjects = len(self.subject_index_map)
        self.n_datasets = len(self.datasets)
        self.n_files = sum(dataset.n_files for dataset in self.datasets)

    def __len__(self):
        return self.n_subjects

    def __getitem__(self, idx):
        d_idx, s_idx = self.subject_index_map[idx]
        return self.datasets[d_idx][s_idx]
