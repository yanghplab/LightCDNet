from dataloaders.datasets import CD_dataset
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):


    train_set = CD_dataset.CDDataset(args, split='train')
    val_set = CD_dataset.CDDataset(args, split='val')
    # test_set = CD_dataset.CDDataset(args, split='test')
    num_class = 2
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    # test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader, val_loader, num_class




