from dataloaders.datasets import CD_dataset
from PIL import Image
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from modeling.LightCDNet import LightCDNet
def get_argparser():
    parser = argparse.ArgumentParser()


    # Datset Options
    parser.add_argument('--isdeconv', action='store_true', default=True,
                        help='whether to use ConvTranspose (default: True)')
    parser.add_argument("--data_root", type=str, default=r'F:\lab\LEVIR-CD',
    # parser.add_argument("--data_root", type=str, default=r'F:\lab\LEVIR-CD',
                        help="path to Dataset")
    parser.add_argument('--base-size', type=int, default=256,
                        help='base image size')
    parser.add_argument("--save_test_results_to", default=r'deeplabv3-LEVIR-CD-256_8_100mobilenet',
                        help="save segmentation results to the specified dir")
    parser.add_argument("--ckpt", default=r'model/LEVIR-CD/ours/model_best.pth.tar', type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument('--masksuffix', type=str, default="png",
                        help='label filename suffix:jpg|png|tif etc.')
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['resnet', 'xception', 'mobilenet'])
    return parser

def main():
    opts = get_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    # Define network
    model = LightCDNet(backbone=opts.backbone, output_stride=16,isdeconv=opts.isdeconv)

    test_set = CD_dataset.CDDataset(opts, split='test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    pretain=torch.load(opts.ckpt, map_location=torch.device('cuda:0'))["state_dict"]
    model_dict = {}
    state_dict = model.state_dict()
    for k, v in pretain.items():
        if k[:8]=='decoder.':
            model_dict[k[8:]] = v
        else:
            model_dict[k] = v
    state_dict.update(model_dict)
    model.load_state_dict(state_dict, strict=False)
    model = nn.DataParallel(model)
    model.to(device)
    if opts.save_test_results_to is not None:
        os.makedirs(opts.save_test_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for image1,image2,target,id in tqdm(test_loader):
            image1, image2, target = image1.cuda(), image2.cuda(), target.cuda()
            pred = model(image1,image2).max(1)[1].cpu().numpy()[0]*255
            pred=pred.astype('uint8')
            colorized_preds = Image.fromarray(pred)
            colorized_preds.save(os.path.join(opts.save_test_results_to, id[0][:-3]+opts.masksuffix ))

import time
if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed = end_time - start_time
    print("train :{}s".format(elapsed/1536))


# 0.07692251602808635s