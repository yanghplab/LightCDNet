import os
import numpy as np
from PIL import Image
from torch.utils import data
from dataloaders.datasets.data_utils import CDDataAugmentation

def default_loader(filename,root1,root2,root3,masksuffix):
    with open(os.path.join(root1, filename), 'rb') as i:
        img1 = Image.open(i).copy()
    with open(os.path.join(root2, filename), 'rb') as i:
        img2 = Image.open(i).copy()
    with open(os.path.join(root3, filename[:-3]+masksuffix), 'rb') as i:
        mask = Image.open(i).copy()
        return img1,img2,mask
class CDDataset(data.Dataset):
    def __init__(self,args, split="train"):
        self.ids = os.listdir(os.path.join(args.data_root,split,"A"))
        self.loader = default_loader
        self.root = args.data_root
        self.split=split
        self.masksuffix=args.masksuffix
        self.img_size=args.base_size
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
        img1,img2, mask = self.loader(id, root1,root2,root3,self.masksuffix)
        img1=np.array(img1,np.uint8)
        img2=np.array(img2,np.uint8)
        mask=np.array(mask,np.uint8)
        label = mask // 255
        [img1, img2], [mask] = self.augm.transform([img1, img2], [label], to_tensor=self.to_tensor)
        return img1, img2, mask, id



    def __len__(self):
        return len(self.ids)


