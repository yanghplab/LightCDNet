from dataloaders.datasets import CD_dataset as cityscapes
from PIL import Image
import argparse
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from modeling.LightCDNet import *
import torchvision.transforms.functional as TF
def get_argparser():
    parser = argparse.ArgumentParser()


    # Datset Options
    parser.add_argument('--isdeconv', type=str, default='deconv',
                        choices=['deconv', 'nodeconv'],
                        help='whether use deconv ')
    parser.add_argument("--data_root", type=str, default=r'F:\lab\LEVIR-CD',
                        help="path to Dataset")
    parser.add_argument("--save_val_results_to", default=r'LEVIR',
                        help="save segmentation results to the specified dir")
    parser.add_argument("--ckpt", default=r'F:\lab\smallpaperresult\code\LightCDNet1\model\LEVIR-CD\ours\model_best.pth.tar', type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument('--backbone', type=str, default='mobilenet',
                        choices=['resnet', 'xception', 'mobilenet'])
    return parser

def main():
    opts = get_argparser().parse_args()



    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    # Define network

    if opts.isdeconv == 'deconv':
        from modeling.LightCDNet import LightCDNet
    if opts.isdeconv == 'nodeconv':
        from modeling.LightCDNet_upsampling import LightCDNet
    model = LightCDNet(num_classes=2,
                       backbone=opts.backbone,
                       output_stride=16,
                       sync_bn=None,
                       freeze_bn=False)
    # Setup dataloader

    test_set = cityscapes.CDDataSet(opts, split='test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["state_dict"],strict=False)
    model = nn.DataParallel(model)
    model.to(device)
    if opts.save_val_results_to is not None:
        os.makedirs(opts.save_val_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for image1,image2,target,id in tqdm(test_loader):
            image1, image2, target = image1.cuda(), image2.cuda(), target.cuda()
            pred = model(image1,image2).max(1)[1].cpu().numpy()[0]*255
            pred=pred.astype('uint8')

            pred = Image.fromarray(pred)

            # pred=TF.resize(pred, [256, 256], interpolation=0)
            # pred=np.array(pred)
            # pred = Image.fromarray(pred)

            pred.save(os.path.join(opts.save_val_results_to, id[0] ))


if __name__ == '__main__':
    # print(1)
    # exit(0)
    main()



