import os
import numpy as np
from PIL import Image
from torch.utils import data
from dataloaders.datasets.data_utils import CDDataAugmentation


def default_loader(filename,root1,root2,root3):
    with open(os.path.join(root1, filename), 'rb') as i:
        img1 = Image.open(i).copy()
    with open(os.path.join(root2, filename), 'rb') as i:
        img2 = Image.open(i).copy()
    with open(os.path.join(root3, filename), 'rb') as i:
        mask = Image.open(i).copy()
    return img1,img2,mask

class CDDataSet(data.Dataset):
    def __init__(self,args, split="train"):
        self.ids = os.listdir(os.path.join(args.data_root,split,"A"))
        self.loader = default_loader
        self.root = args.data_root
        self.split=split
        self.img_size=256
        self.to_tensor=True
        if self.split=="train":

            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
                random_color_tf=True
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )
    def __getitem__(self, index):
        id = self.ids[index]
        root1=os.path.join(self.root,self.split,"A")
        root2=os.path.join(self.root,self.split,"B")
        root3=os.path.join(self.root,self.split,"label")

        img1,img2, mask = self.loader(id, root1,root2,root3)
        img1=np.array(img1,np.uint8)
        img2=np.array(img2,np.uint8)
        mask=np.array(mask,np.uint8)
        label = mask // 255
        [img1, img2], [mask] = self.augm.transform([img1, img2], [label], to_tensor=self.to_tensor)
        return img1, img2, mask, id
    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 513
    args.crop_size = 513

    cityscapes_train = CDDataSet(args, split='train')

    dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

        if ii == 1:
            break

    plt.show(block=True)

