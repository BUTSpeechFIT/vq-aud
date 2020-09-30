import torch
from torch.utils.data import Dataset, DataLoader


class SimpleDataset(Dataset):
    def __init__(self,
                 data,
                 docs,
                 pad_value=-1000,
                 sort_by_length=True):
        self.data = data
        self.docs = docs
        self.pad_value = pad_value
        lens = [len(self.data[x]) for x in self.docs]
        self.max_length_frames = max(lens)
        if sort_by_length:
            self.docs = sorted(self.docs, key=lambda x: len(self.data[x]))

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, ind):
        data = torch.from_numpy(self.data[self.docs[ind]])
        return data, data


def batchify(all_samples, pad_value=-1000):
    lens = [len(x[0]) for x in all_samples]
    max_length_frames = max(lens)
    data_batch = torch.zeros(len(all_samples), max_length_frames, all_samples[0][0].shape[-1])
    lens = [len(x[1]) for x in all_samples]
    max_length_frames = max(lens)
    label_batch = pad_value * torch.ones(len(all_samples), max_length_frames, all_samples[0][1].shape[-1])
    for i, (sample, label) in enumerate(all_samples):
        data_batch[i, :len(sample)] = sample
        label_batch[i, :len(label)] = label
    return data_batch, label_batch


def dataloader(dataset_kind=SimpleDataset,
               batch_size=512,
               num_workers=10,
               shuffle=False,
               pin_memory=True,
               **dataset_kwargs):
    dataset = dataset_kind(**dataset_kwargs)
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      collate_fn=batchify,
                      )


_dataset_names = {'SimpleDataset': SimpleDataset,
                  }


def get_dataset(dataset_name):
    return _dataset_names[dataset_name]
