from torch.utils.data import DataLoader
from utils.Dataset import Dataset

# from Dataset import Dataset

def build_dataloader(batch_size:int, num_workers:int, index:int, image_name:str, **kwargs):
    return DataLoader(
        dataset=Dataset(image_name=image_name, index=index, **kwargs),
        batch_size=batch_size,
        num_workers=num_workers,
        # shuffle=True, 在Dataset手动shuffle
    )

if __name__ == '__main__':
    dataloader = build_dataloader(
        mode='val',
        batch_size=4,
        num_workers=0,
        dataset_dir_paths=['./DamDataset']
    )
    for seqs, labels in dataloader:
        print(seqs)
        print(labels)