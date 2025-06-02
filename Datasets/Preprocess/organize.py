import os
import rarfile


class AbstractRfMRIDataset:
    def __init__(self, src, dst, filenames, dataset_name=None, phenotype=None):
        self.src = src
        self.dst = dst
        self.filenames = filenames
        self.files = [os.path.join(self.src, file) for file in self.filenames]
        self.dataset_name = dataset_name
        self.phenotype = phenotype or self._detect_phenotype_file()
        self.file2uncompress = self._detect_files_to_uncompress()

    def _detect_phenotype_file(self):
        for f in self.files:
            if f.endswith(".csv"):
                return f
        raise ValueError("❌ Could not find a phenotype (.csv) file in the given filenames.")

    def _detect_files_to_uncompress(self):
        """Return only the top-level .rar or .part1.rar files for extraction."""
        files_to_uncompress = []

        for file in self.files:
            if file.endswith('.part1.rar'):
                files_to_uncompress.append(file)
            elif file.endswith('.rar') and '.part' not in file:
                files_to_uncompress.append(file)

        if not files_to_uncompress:
            raise ValueError("❌ No valid .rar or .part1.rar files found for extraction.")
        
        return files_to_uncompress

    def check_files(self):
        for file in self.files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"❌ File not found: {file}")
        print(f"✅ All files are present for {self.dataset_name} dataset.")

    def uncompress(self):
        for file in self.file2uncompress:
            print(f"📦 Extracting {os.path.basename(file)}...")
            with rarfile.RarFile(file) as rf:
                rf.extractall(self.dst)
                print(f"✅ Extracted to: {self.dst}")

    def organize(self):
        self.check_files()
        self.uncompress()


if __name__ == "__main__":
    abide1 = AbstractRfMRIDataset(
        src='/datastorage/li/fMRI/ABIDE',
        dst='/datastorage/li/fMRI/ABIDE',
        filenames=[
            'ABIDE_RfMRI.part1.rar', 'ABIDE_RfMRI.part2.rar', 'ABIDE_RfMRI.part3.rar',
            'RfMRIMaps_ABIDE_Phenotypic.csv'
        ],
        dataset_name='ABIDE1'
    )
    abide1.organize()
