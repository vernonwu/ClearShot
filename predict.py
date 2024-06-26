import os
import torch
import argparse
from basicsr.models.archs.adaptive_fftformer import Adaptive_FFTFormer
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from torchvision.transforms import functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from thop import profile
from PIL import Image as Image
from tqdm import tqdm
import numpy as np
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
import threading
import queue


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, require_label=True):
        self.image_dir = "/root/projects/FFTformer/media/pred/"
        self.datasets = os.listdir(os.path.join(image_dir, 'input'))
        self.image_list = []
        for dataset in self.datasets:
            image_list = os.listdir(os.path.join(image_dir, 'input', dataset))
            self.image_list += [os.path.join(dataset, x) for x in image_list if x !=".ipynb_checkpoints"]
        print(self.image_list)
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.require_label = require_label

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'input', self.image_list[idx]))
        if self.require_label:
            label = Image.open(os.path.join(self.image_dir, 'target', self.image_list[idx]))
        else:
            label = None

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            if label is not None:
                label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            if not self.require_label:
                return image, name
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError


def test_dataloader(path, batch_size=1, num_workers=0, require_label=True):
    image_dir = path
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True, require_label=require_label),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def _eval(model, data_dir, result_dir, pred=True, save_image=True):
    device = torch.device('cuda')
    dataloader = test_dataloader(data_dir, batch_size=1, num_workers=8, require_label=not pred)
    torch.cuda.empty_cache()

    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():
        for iter_idx, data in tqdm(enumerate(dataloader)):
            if pred:
                input_img, name = data
            else:
                input_img, label_img, name = data

            input_img = input_img.to(device)

            b, c, h, w = input_img.shape
            h_n = (32 - h % 32) % 32
            w_n = (32 - w % 32) % 32
            input_img = torch.nn.functional.pad(input_img, (0, w_n, 0, h_n), mode='reflect')

            pred_img = model(input_img)
            torch.cuda.synchronize()
            pred_img = pred_img[:, :, :h, :w]

            pred_clip = torch.clamp(pred_img, 0, 1)
            if not pred:
                label_img = label_img.to(device)
                crop_border = 4
                psnr = calculate_psnr(label_img, pred_clip, crop_border=crop_border)
                ssim = calculate_ssim(label_img, pred_clip, crop_border=crop_border)

                psnr_scores.append(psnr)
                ssim_scores.append(ssim)

                print(f'index {iter_idx} : psnr={psnr}, ssim={ssim}')

            if save_image:
                dataset = name[0].split('/')[0]
                if not os.path.exists(os.path.join(result_dir, dataset)):
                    os.makedirs(os.path.join(result_dir, dataset))
                save_name = os.path.join(result_dir, name[0])
                save_name = "/root/projects/FFTformer/"+save_name
                pred_clip += 0.5 / 255
                pred_img = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred_img.save(save_name)

    if not pred:
        avg_psnr = np.mean(psnr_scores)
        avg_ssim = np.mean(ssim_scores)

        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average SSIM: {avg_ssim:.4f}")

def _eval2(model, q, result_dir,device_num='0', pred=True, save_image=True):

    device = torch.device('cuda:'+device_num)
    torch.cuda.empty_cache()

    psnr_scores = []
    ssim_scores = []
    image_dir = "/root/projects/FFTformer/media/pred/"
    transform = transforms.Compose([transforms.ToTensor()])
    with torch.no_grad():
        while True:
            if not q.empty():
                start_time = time.time()
                if pred:
                    name = q.get()
                    while 1:
                        try:
                            input_img = Image.open("/root/projects/FFTformer/media/pred/input/1/"+ name)
                            input_img = transform(input_img).unsqueeze(0)
                        except Exception:
                            continue
                        else:
                            break
                else:
                    input_img, label_img, name = data
                    

                input_img = input_img.to(device)
        
                b, c, h, w = input_img.shape
                h_n = (128 - h % 128) % 128
                w_n = (128 - w % 128) % 128
                input_img = torch.nn.functional.pad(input_img, (0, w_n, 0, h_n), mode='reflect')
        
                pred_img = model(input_img)
                torch.cuda.synchronize()
                pred_img = pred_img[:, :, :h, :w]
        
                pred_clip = torch.clamp(pred_img, 0, 1)
                if not pred:
                    label_img = label_img.to(device)
                    crop_border = 4
                    psnr = calculate_psnr(label_img, pred_clip, crop_border=crop_border)
                    ssim = calculate_ssim(label_img, pred_clip, crop_border=crop_border)
        
                    psnr_scores.append(psnr)
                    ssim_scores.append(ssim)
        
                    print(f'index {iter_idx} : psnr={psnr}, ssim={ssim}')
        
                if save_image:
                    save_name = "projects/FFTformer/results/Adaptive_fftformer/output/1/"+name
                    pred_clip += 0.5 / 255
                    pred_img = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                    pred_img.save(save_name)
                    end_time = time.time()
                    print("Image:{} done, cost time{}".format(name,end_time-start_time))
        
            if not pred:
                avg_psnr = np.mean(psnr_scores)
                avg_ssim = np.mean(ssim_scores)
        
                print(f"Average PSNR: {avg_psnr:.2f}")
                print(f"Average SSIM: {avg_ssim:.4f}")
            time.sleep(0.1)


class NewImageHandler(FileSystemEventHandler):
    def __init__(self, model, args):
        self.model = model
        self.args = args

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.png', '.jpg', '.jpeg')):
            print(f'New image detected: {event.src_path}')
            while True:
                try:
                    _eval(self.model, self.args.data_dir, self.args.result_dir, self.args.pred, self.args.save_image)
                except Exception:
                    continue
                else:
                    break
"""            
            folder_path = os.path.join(self.args.data_dir,'input','1')
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f'无法删除文件: {file_path}. 错误信息: {e}')
            print("done")
            """

class put_queue_handler(FileSystemEventHandler):
    def __init__(self, q): 
        self.q = q
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith(('.png', '.jpg', '.jpeg')):
            print(f'New image detected: {event.src_path}')
            while True:
                try:
                    self.q.put(event.src_path.split('/')[-1])
                except Exception:
                    continue
                else:
                    break


def main(args):
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model_gpu0 = Adaptive_FFTFormer(pretrained=args.test_model)
    model_gpu1 = Adaptive_FFTFormer(pretrained=args.test_model)
    if torch.cuda.is_available():
        model_gpu0.cuda(0)
        model_gpu1.cuda(1)

    q = queue.Queue()
    event_handler = put_queue_handler(q)
    observer = Observer()
    observer.schedule(event_handler, path=args.data_dir, recursive=True)
    thread_gpu0 = threading.Thread(target=_eval2, args=(model_gpu0,q,args.result_dir,'0',))
    thread_gpu1 = threading.Thread(target=_eval2, args=(model_gpu1,q,args.result_dir,'1',))
    thread_gpu0.start()
    thread_gpu1.start()
    observer.start()
     
    observer.join()
    thread_gpu0.join()
    thread_gpu1.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='Adaptive_fftformer', type=str)
    parser.add_argument('--data_dir', type=str, default='projects/FFTformer/media/pred/')

    parser.add_argument('--test_model', type=str, default='projects/FFTformer/pretrain_model/net_g_Realblur_J.pth')
    parser.add_argument('--pred', type=bool, default=True, choices=[True, False])
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

    args = parser.parse_args()
    args.result_dir = os.path.join('results/', args.model_name, 'output')
    print(args)
    main(args)