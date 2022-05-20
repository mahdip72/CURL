import matplotlib
import numpy as np
import sys
import os
import torch
import model
import lpips
import matplotlib.pyplot as plt
from training.pytorch.curl.utils import calculate_metrics
from torch.utils.tensorboard import SummaryWriter
from utils import torch_psnr, torch_msssim, torch_sam, torch_lpips, pcqi_metric, torch_ssim
from torchvision import transforms

matplotlib.use('agg')
np.set_printoptions(threshold=sys.maxsize)


class Evaluator:
    def __init__(self, data_loader, set_name, save_path, device, log=True, plot=False):
        super().__init__()
        self.data_loader = data_loader
        self.set_name = set_name
        self.save_path = save_path
        self.image_path = os.path.join(save_path, set_name, 'images')
        self.writer = SummaryWriter(os.path.join(save_path, set_name, 'tensorboard_logs'))
        self.device = device
        self.plot = plot
        self.log = log
        net = 'vgg'
        self.lpips_model = lpips.LPIPS(net=net, version=0.1).to(device)
        os.mkdir(self.image_path)

    @staticmethod
    def normalizing_image(img):
        norm = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        return norm.forward(img)

    def compute_metrics(self, pred_img, target_img):
        pred_img = pred_img[None, ...].to(self.device)
        target_img = target_img[None, ...].to(self.device)

        normed_pred_img = self.normalizing_image(pred_img).to(self.device)
        normed_target_img = self.normalizing_image(target_img).to(self.device)

        msssim = torch_msssim(pred_img, target_img)
        ssim = torch_ssim(pred_img, target_img)
        psnr = torch_psnr(pred_img, target_img)
        lpips_value = torch_lpips(self.lpips_model, normed_pred_img, normed_target_img)
        sam = torch_sam(pred_img, target_img)

        pcqi = pcqi_metric(pred_img*255, target_img*255, self.device)
        return ssim, msssim, psnr, lpips_value, sam, pcqi

    def evaluate(self, net, epoch=0):
        """Evaluates a network on a specified split of a dataset e.g. test, validation
        """
        if not (self.log or self.plot):
            return 0, 0

        examples = 0
        num_batches = len(self.data_loader)
        initial_device = net.device
        net.device = self.device
        net.to(self.device)
        net.eval()
        with torch.no_grad():
            mean_psnr = torch.zeros(1).to(self.device)
            mean_l1_lab = torch.zeros(1).to(self.device)
            mean_l1_rgb = torch.zeros(1).to(self.device)
            mean_ssim = torch.zeros(1).to(self.device)
            mean_ms_ssim = torch.zeros(1).to(self.device)
            mean_curl_ms_ssim = torch.zeros(1).to(self.device)
            mean_sam = torch.zeros(1).to(self.device)
            mean_pcqi = torch.zeros(1).to(self.device)
            mean_lpips_value = torch.zeros(1).to(self.device)

            for batch_num, (input_img_batch, gt_img_batch, name) in enumerate(self.data_loader):
                input_img_batch = input_img_batch.to(self.device)
                gt_img_batch = gt_img_batch.to(self.device)

                img_name = name[0].split('\\')[-1][:-4]
                pred_img, _ = net(input_img_batch)

                if self.plot:
                    output_img_example = (
                            torch.clamp(pred_img[0].permute(1, 2, 0), 0, 1) * 255).cpu().numpy().astype(
                        'uint8')

                    image_dir = os.path.join(self.image_path,
                                             img_name + "_" + "_" + str(epoch + 1) + "_" + str(examples) + ".jpg")
                    plt.imsave(image_dir, output_img_example)

                    self.writer.add_image('Results on ' + img_name, pred_img[0], epoch,
                                          dataformats='CHW')

                if self.log:
                    l1_lab, l1_rgb, curl_msssim = calculate_metrics(pred_image=pred_img[0],
                                                                    target_image=gt_img_batch[0], device=self.device)
                    ssim, msssim, psnr, lpips_value, sam, pcqi = self.compute_metrics(pred_img[0], gt_img_batch[0])

                    mean_psnr += psnr
                    mean_l1_lab += l1_lab
                    mean_l1_rgb += l1_rgb
                    mean_ssim += ssim
                    mean_ms_ssim += msssim
                    mean_curl_ms_ssim += curl_msssim
                    mean_sam += sam
                    mean_lpips_value += lpips_value
                    mean_pcqi += pcqi

            if self.log:
                mean_psnr /= num_batches
                mean_l1_lab /= num_batches
                mean_l1_rgb /= num_batches
                mean_ssim /= num_batches
                mean_ms_ssim /= num_batches
                mean_curl_ms_ssim /= num_batches
                mean_sam /= num_batches
                mean_lpips_value /= num_batches
                mean_pcqi /= num_batches

                self.writer.add_scalar("l1_rgb", mean_l1_rgb, epoch)
                self.writer.add_scalar("l1_lab", mean_l1_lab, epoch)
                self.writer.add_scalar("curl_msssim", mean_curl_ms_ssim, epoch)
                self.writer.add_scalar("msssim", mean_ms_ssim, epoch)
                self.writer.add_scalar("ssim", mean_ssim, epoch)
                self.writer.add_scalar("sam", mean_sam, epoch)
                self.writer.add_scalar("psnr", mean_psnr, epoch)
                self.writer.add_scalar("lpips", mean_lpips_value, epoch)
                self.writer.add_scalar("pcqi", mean_pcqi, epoch)
                self.writer.flush()

        net.device = initial_device
        net.to(initial_device)

        return round(mean_psnr.cpu().numpy()[0], 4), round(mean_ssim.cpu().numpy()[0], 4)
