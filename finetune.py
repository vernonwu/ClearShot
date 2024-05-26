import os
import torch
import argparse
from basicsr.models.archs.adaptive_fftformer import Adaptive_FFTFormer
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from thop import profile
from PIL import Image as Image
from tqdm import tqdm
import numpy as np


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.datasets = os.listdir(os.path.join(image_dir, 'input'))
        self.image_list = []
        for dataset in self.datasets:
            image_list = os.listdir(os.path.join(image_dir, 'input', dataset))
            self.image_list += [os.path.join(dataset, x) for x in image_list]
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'input', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'target', self.image_list[idx]))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader



def main(args):
    # CUDNN
    # cudnn.benchmark = True
    #
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = Adaptive_FFTFormer(pretrained=args.test_model)
    print(model)
    if torch.cuda.is_available():
        model.cuda()

    _eval(model, args)

def _eval(model, args):

    device = torch.device('cuda')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=8)
    torch.cuda.empty_cache()

    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():

        # Main Evaluation
        for iter_idx, data in tqdm(enumerate(dataloader)):
            input_img, label_img, name = data

            input_img = input_img.to(device)

            b, c, h, w = input_img.shape
            h_n = (32 - h % 32) % 32
            w_n = (32 - w % 32) % 32
            input_img = torch.nn.functional.pad(input_img, (0, w_n, 0, h_n), mode='reflect')

            pred = model(input_img)
            torch.cuda.synchronize()
            pred = pred[:, :, :h, :w]

            pred_clip = torch.clamp(pred, 0, 1)

            # Calculate PSNR
            label_img = label_img.to(device)
            crop_border = 4
            psnr = calculate_psnr(label_img, pred_clip,crop_border=crop_border)
            psnr_scores.append(psnr)
            # Calculate SSIM
            ssim = calculate_ssim(label_img, pred_clip,crop_border=crop_border)
            ssim_scores.append(ssim)

            print(f'iter {iter_idx} : psnr={psnr}, ssim={ssim}')

            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)

            # flops, params = profile(model, inputs=(input_img,), verbose=False)
            # print(f"FLOPs: {flops / 1e9:.2f} G")
            # print(f"Parameter Size: {params / 1e6:.2f} M")

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)

    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='fftformer', type=str)
    parser.add_argument('--data_dir', type=str, default='./media/val/')

    # Test
    parser.add_argument('--test_model', type=str, default='./pretrain_model/net_g_Realblur_J.pth')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.result_dir = os.path.join('results/', args.model_name, 'GoPro/')
    print(args)
    main(args)
