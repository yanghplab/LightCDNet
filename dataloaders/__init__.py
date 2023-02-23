from dataloaders.datasets import CD_dataset as cityscapes
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):


    train_set = cityscapes.CDDataSet(args, split='train')
    num_class = 2
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_set = cityscapes.CDDataSet(args, split='val')
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader, num_class


