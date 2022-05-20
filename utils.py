import numpy as np
import torchvision.transforms as transforms
from math import exp
from PIL import Image
import torch.nn.functional as F
import os
import math
import torch
import sys
import model
import logging
import cv2
import matplotlib
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import multiscale_structural_similarity_index_measure, peak_signal_noise_ratio
from torchmetrics.functional import spectral_angle_mapper, structural_similarity_index_measure
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from adabelief_pytorch import AdaBelief
from skimage.metrics import structural_similarity as ssim
from torch.autograd import Variable

matplotlib.use('agg')
np.set_printoptions(threshold=sys.maxsize)
alpha = 0.5
ssim_window_size = 5


def test_gpu_cuda():
    print('Testing gpu and cuda:')
    print('\tcuda is available:', torch.cuda.is_available())
    print('\tdevice count:', torch.cuda.device_count())
    print('\tcurrent device:', torch.cuda.current_device())
    print(f'\tdevice:', torch.cuda.device(0))
    print('\tdevice name:', torch.cuda.get_device_name(), end='\n\n')


def compute_pcqi(img1, img2, device, L=256.):
    window = matlab_style_gauss2D(shape=(11, 11), sigma=1.5).to(device)

    window = window / torch.sum(window)
    mul = torch.conv2d(img1, window)
    mul2 = torch.conv2d(img2, window)

    mul_sq = torch.mul(mul, mul)
    mul2_sq = torch.mul(mul2, mul2)
    mul_mul2 = torch.mul(mul, mul2)

    sigma1_sq = torch.conv2d(torch.mul(img1 / 100., img1 / 100.) * 1e4, window) - mul_sq

    sigma2_sq = torch.conv2d(torch.mul(img2 / 100., img2 / 100.) * 1e4, window) - mul2_sq
    sigma12 = torch.conv2d(torch.mul(img1 / 100., img2 / 100.) * 1e4, window) - mul_mul2

    sigma1_sq[sigma1_sq < 0] = 0
    sigma2_sq[sigma2_sq < 0] = 0

    C = 3
    pcqi_map = (4. / torch.tensor(np.pi).to(device)) * torch.arctan(torch.divide(sigma12 + C, sigma1_sq + C))

    pcqi_map = torch.mul(pcqi_map, torch.divide(sigma12 + C, torch.mul(torch.sqrt(sigma1_sq) / 100.0,
                                                                       torch.sqrt(sigma2_sq) / 100.0) * 1e4 + C))

    pcqi_map = torch.mul(pcqi_map, torch.exp(-torch.abs(mul - mul2) / L))

    mpcqi = torch.mean(pcqi_map)

    return mpcqi, pcqi_map


def pcqi_metric(img1, img2, device):
    """
    prints the outputs of PCQI metric for corresponding images.
    1 means the second image is just like the first one
    above 1 means first image is better
    below 1 means first image in worse
    """
    transform = transforms.Grayscale()
    img1 = transform(img1.squeeze().double())
    img2 = transform(img2.squeeze().double())

    score, p_map = compute_pcqi(img1[None, ...], img2[None, ...], device)
    return round(score.item(), 4)


def torch_lpips(lpips_model, img1, img2):
    return round(lpips_model(img1, img2).item(), 4)


def torch_ssim(img1, img2):
    return round(structural_similarity_index_measure(img1, img2).item(), 4)


def torch_msssim(img1, img2):
    return round(multiscale_structural_similarity_index_measure(img1, img2).item(), 4)


def torch_psnr(img1, img2):
    return round(peak_signal_noise_ratio(img1, img2).item(), 4)


def torch_sam(img1, img2):
    img1[img1 == 0] = 1e-6
    img2[img2 == 0] = 1e-6
    return round(spectral_angle_mapper(img1, img2).item(), 4)


def prepare_logging(save_path, train_path, valid_path, test_path):
    handlers = [logging.FileHandler(os.path.join(save_path, "log.txt")), logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)

    logging.info('Logging directory: ' + str(save_path))
    logging.info('Training image directory: ' + str(train_path))
    logging.info('Validation image directory: ' + str(valid_path))
    logging.info('Test image directory: ' + str(test_path))


def prepare_model(device, pretrained_weights, print_model=False):
    net = model.CURLNet(device=device)
    net = net.to(device)
    logging.info('######### Network created #########')
    logging.info(f'Number of parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}')

    if print_model:
        logging.info('Architecture:\n' + str(net))

    # for name, param in net.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    if pretrained_weights:
        logging.info('Load pretrained weights')
        checkpoint = torch.load(pretrained_weights, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
    else:
        logging.info('train using random weights')

    return net


def prepare_optimizer(name, net, lr, min_lr, train_samples, batch, epochs, warmup, gamma, wd,
                      weight_decouple=False, eps=1e-08):
    if name.lower() == 'adam' and weight_decouple:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                                      lr=lr, eps=eps, weight_decay=wd)
    elif name.lower() == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                     lr=lr, eps=eps, weight_decay=wd)
    elif name.lower() == 'adabelief':
        optimizer = AdaBelief(filter(lambda p: p.requires_grad, net.parameters()), lr=lr,
                              weight_decouple=weight_decouple, weight_decay=wd,
                              fixed_decay=False, eps=eps, rectify=False, print_change_log=False)
    else:
        logging.info('wrong given optimizer:', name)
        optimizer = None
        exit()

    logging.info(f'select {name} optimizer')

    epoch_steps = int(np.ceil(train_samples / batch))
    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                 first_cycle_steps=epoch_steps * epochs,
                                                 cycle_mult=1.0,
                                                 max_lr=lr,
                                                 min_lr=min_lr,
                                                 warmup_steps=int(epoch_steps * warmup),
                                                 gamma=gamma)
    logging.info(f'select cosine lr scheduler with {warmup} epochs warmup')

    return optimizer, lr_scheduler


def calculate_metrics(pred_image, target_image, device):
    predicted_img_lab = torch.clamp(rgb_to_lab(pred_image.squeeze(0), device), 0, 1)
    target_img_lab = torch.clamp(rgb_to_lab(target_image.squeeze(0), device), 0, 1)
    l1_lab = F.l1_loss(predicted_img_lab, target_img_lab)
    l1_rgb = F.l1_loss(pred_image.to(torch.float32), target_image.to(torch.float32))
    # psnr = compute_psnr(pred_image, target_image)

    target_img_l_ssim = target_img_lab[0, :, :].unsqueeze(0)
    predicted_img_l_ssim = predicted_img_lab[0, :, :].unsqueeze(0)
    target_img_l_ssim = target_img_l_ssim.unsqueeze(0)
    predicted_img_l_ssim = predicted_img_l_ssim.unsqueeze(0)
    ssim_value = compute_msssim(predicted_img_l_ssim, target_img_l_ssim, device)

    # ssim_loss_value = (1.0 - ssim_value)

    return l1_lab, l1_rgb, ssim_value


def rgb_to_lab(img, device, is_training=True):
    img = img.permute(2, 1, 0)
    shape = img.shape
    img = img.contiguous()
    img = img.view(-1, 3)
    img = (img / 12.92) * img.le(0.04045).float() + (((torch.clamp(img,
                                                                   min=0.0001) + 0.055) / 1.055) ** 2.4) * img.gt(
        0.04045).float()

    rgb_to_xyz = Variable(torch.FloatTensor([  # X        Y          Z
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169,
         0.950227],  # B
    ]), requires_grad=False).to(device)

    img = torch.matmul(img, rgb_to_xyz)
    img = torch.mul(img, Variable(torch.FloatTensor([1 / 0.950456, 1.0, 1 / 1.088754]), requires_grad=False).to(device))

    epsilon = 6 / 29

    img = ((img / (3.0 * epsilon ** 2) + 4.0 / 29.0) * img.le(epsilon ** 3).float()) + \
          (torch.clamp(img, min=0.0001) **
           (1.0 / 3.0) * img.gt(epsilon ** 3).float())

    fxfyfz_to_lab = Variable(torch.FloatTensor([[0.0, 500.0, 0.0],  # fx
                                                # fy
                                                [116.0, -500.0, 200.0],
                                                # fz
                                                [0.0, 0.0, -200.0],
                                                ]), requires_grad=False).to(device)

    img = torch.matmul(img, fxfyfz_to_lab) + Variable(
        torch.FloatTensor([-16.0, 0.0, 0.0]), requires_grad=False).to(device)

    img = img.view(shape)
    img = img.permute(2, 1, 0)

    img[0, :, :] = img[0, :, :] / 100
    img[1, :, :] = (img[1, :, :] / 110 + 1) / 2
    img[2, :, :] = (img[2, :, :] / 110 + 1) / 2

    img[(img != img).detach()] = 0

    img = img.contiguous()
    return img.to(device)


def load_image(img_filepath, normaliser):
    img = normalise_image(np.array(Image.open(img_filepath)),
                          normaliser)  # NB: imread normalises to 0-1
    return img


def normalise_image(img, normaliser):
    img = img.astype('float32') / normaliser
    return img


def rgb_to_hsv(img, device):
    img = torch.clamp(img, 1e-9, 1)
    img = img.permute(2, 1, 0)
    shape = img.shape

    img = img.contiguous()
    img = img.view(-1, 3)

    mx = torch.max(img, 1)[0]
    mn = torch.min(img, 1)[0]

    ones = Variable(torch.FloatTensor(torch.ones((img.shape[0])))).to(device)
    zero = Variable(torch.FloatTensor(torch.zeros(shape[0:2]))).to(device)

    img = img.view(shape)

    ones1 = ones[0:math.floor((ones.shape[0] / 2))]
    ones2 = ones[math.floor(ones.shape[0] / 2):(ones.shape[0])]

    mx1 = mx[0:math.floor((ones.shape[0] / 2))]
    mx2 = mx[math.floor(ones.shape[0] / 2):(ones.shape[0])]
    mn1 = mn[0:math.floor((ones.shape[0] / 2))]
    mn2 = mn[math.floor(ones.shape[0] / 2):(ones.shape[0])]

    df1 = torch.add(mx1, torch.mul(ones1 * -1, mn1))
    df2 = torch.add(mx2, torch.mul(ones2 * -1, mn2))

    df = torch.cat((df1, df2), 0)
    del df1, df2
    df = df.view(shape[0:2]) + 1e-10
    mx = mx.view(shape[0:2])

    img = img.to(device)
    df = df.to(device)
    mx = mx.to(device)

    g = img[:, :, 1].clone().to(device)
    b = img[:, :, 2].clone().to(device)
    r = img[:, :, 0].clone().to(device)

    img_copy = img.clone()

    img_copy[:, :, 0] = (((g - b) / df) * r.eq(mx).float() + (2.0 + (b - r) / df)
                         * g.eq(mx).float() + (4.0 + (r - g) / df) * b.eq(mx).float())
    img_copy[:, :, 0] = img_copy[:, :, 0] * 60.0

    zero = zero.to(device)
    img_copy2 = img_copy.clone()

    img_copy2[:, :, 0] = img_copy[:, :, 0].lt(zero).float(
    ) * (img_copy[:, :, 0] + 360) + img_copy[:, :, 0].ge(zero).float() * (img_copy[:, :, 0])

    img_copy2[:, :, 0] = img_copy2[:, :, 0] / 360

    del img, r, g, b

    img_copy2[:, :, 1] = mx.ne(zero).float() * (df / mx) + \
                         mx.eq(zero).float() * (zero)
    img_copy2[:, :, 2] = mx

    img_copy2[(img_copy2 != img_copy2).detach()] = 0

    img = img_copy2.clone()

    img = img.permute(2, 1, 0)
    img = torch.clamp(img, 1e-9, 1)
    return img


def create_window(window_size, num_channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(num_channel, 1, window_size, window_size).contiguous())
    return window


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def compute_ssim(img1, img2, device):
    (_, num_channel, _, _) = img1.size()
    window = create_window(ssim_window_size, num_channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
        window = window.type_as(img1)

    mu1 = F.conv2d(
        img1, window, padding=ssim_window_size // 2, groups=num_channel)
    mu2 = F.conv2d(
        img2, window, padding=ssim_window_size // 2, groups=num_channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=ssim_window_size // 2, groups=num_channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=ssim_window_size // 2, groups=num_channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=ssim_window_size // 2, groups=num_channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map1 = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    ssim_map2 = ((mu1_sq.to(device) + mu2_sq.to(device) + C1) *
                 (sigma1_sq.to(device) + sigma2_sq.to(device) + C2))
    ssim_map = ssim_map1.to(device) / ssim_map2.to(device)

    v1 = 2.0 * sigma12.to(device) + C2
    v2 = sigma1_sq.to(device) + sigma2_sq.to(device) + C2
    cs = torch.mean(v1 / v2)

    return ssim_map.mean(), cs


def compute_psnr(imageA, imageB):
    imageA = imageA.cpu().numpy()
    imageB = imageB.cpu().numpy()
    psnr_val = 10 * np.log10(1 / ((imageA - imageB) ** 2).mean())
    return psnr_val


def compute_msssim(img1, img2, device):
    if img1.shape[2] != img2.shape[2]:
        img1 = img1.transpose(2, 3)

    if img1.shape != img2.shape:
        print(img1.shape, img2.shape)
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)

    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    # device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []
    for _ in range(levels):
        ssim, cs = compute_ssim(img1, img2, device)

        ssims.append(ssim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)

    ssims = (ssims + 1) / 2
    mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = ssims ** weights

    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


def matlab_style_gauss2D(shape=(11, 11), sigma=1.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    h = h[::-1, ::-1].copy()
    return torch.from_numpy(h)[None, None, :, :]


class ImageProcessing(object):

    @staticmethod
    def rgb_to_lab(img, device, is_training=True):
        img = img.to(device)
        img = img.permute(0, 3, 2, 1)
        shape = img.shape
        img = img.contiguous()
        img = img.view(img.shape[0], -1, 3)
        img = (img / 12.92) * img.le(0.04045).float() + (((torch.clamp(img,
                                                                       min=0.0001) + 0.055) / 1.055) ** 2.4) * img.gt(
            0.04045).float()

        rgb_to_xyz = Variable(torch.FloatTensor([  # X        Y          Z
            [0.412453, 0.212671, 0.019334],  # R
            [0.357580, 0.715160, 0.119193],  # G
            [0.180423, 0.072169,
             0.950227],  # B
        ]), requires_grad=False).to(device)

        img = torch.matmul(img, rgb_to_xyz)
        img = torch.mul(img, Variable(torch.FloatTensor(
            [1 / 0.950456, 1.0, 1 / 1.088754]), requires_grad=False).to(device))

        epsilon = 6 / 29

        img = ((img / (3.0 * epsilon ** 2) + 4.0 / 29.0) * img.le(epsilon ** 3).float()) + \
              (torch.clamp(img, min=0.0001) **
               (1.0 / 3.0) * img.gt(epsilon ** 3).float())

        fxfyfz_to_lab = Variable(torch.FloatTensor([[0.0, 500.0, 0.0],  # fx
                                                    # fy
                                                    [116.0, -500.0, 200.0],
                                                    # fz
                                                    [0.0, 0.0, -200.0],
                                                    ]), requires_grad=False).to(device)

        img = torch.matmul(img, fxfyfz_to_lab) + Variable(
            torch.FloatTensor([-16.0, 0.0, 0.0]), requires_grad=False).to(device)

        img = img.view(shape)
        img = img.permute(0, 3, 2, 1)

        img[:, 0, :, :] = img[:, 0, :, :] / 100
        img[:, 1, :, :] = (img[:, 1, :, :] / 110 + 1) / 2
        img[:, 2, :, :] = (img[:, 2, :, :] / 110 + 1) / 2

        img[(img != img).detach()] = 0

        img = img.contiguous()
        return img.to(device)

    @staticmethod
    def lab_to_rgb(img, device, is_training=True):
        img = img.permute(0, 3, 2, 1)
        shape = img.shape
        img = img.contiguous()
        img = img.view(img.shape[0], -1, 3)
        img_copy = img.clone()

        img_copy[:, :, 0] = img[:, :, 0] * 100
        img_copy[:, :, 1] = ((img[:, :, 1] * 2) - 1) * 110
        img_copy[:, :, 2] = ((img[:, :, 2] * 2) - 1) * 110

        img = img_copy.clone().to(device)
        del img_copy

        lab_to_fxfyfz = Variable(torch.FloatTensor([  # X Y Z
            [1 / 116.0, 1 / 116.0, 1 / 116.0],  # R
            [1 / 500.0, 0, 0],  # G
            [0, 0, -1 / 200.0],  # B
        ]), requires_grad=False).to(device)

        img = torch.matmul(img + Variable(torch.FloatTensor([16.0, 0.0, 0.0])).to(device), lab_to_fxfyfz)

        epsilon = 6.0 / 29.0

        img = (((3.0 * epsilon ** 2 * (img - 4.0 / 29.0)) * img.le(epsilon).float()) +
               ((torch.clamp(img, min=0.0001) ** 3.0) * img.gt(epsilon).float()))

        # denormalize for D65 white point
        img = torch.mul(img, Variable(
            torch.FloatTensor([0.950456, 1.0, 1.088754])).to(device))

        xyz_to_rgb = Variable(torch.FloatTensor([  # X Y Z
            [3.2404542, -0.9692660, 0.0556434],  # R
            [-1.5371385, 1.8760108, -0.2040259],  # G
            [-0.4985314, 0.0415560, 1.0572252],  # B
        ]), requires_grad=False).to(device)

        img = torch.matmul(img, xyz_to_rgb)

        img = (img * 12.92 * img.le(0.0031308).float()) + ((torch.clamp(img,
                                                                        min=0.0001) ** (
                                                                    1 / 2.4) * 1.055) - 0.055) * img.gt(
            0.0031308).float()

        img = img.view(shape)
        img = img.permute(0, 3, 2, 1)

        img = img.contiguous()
        img[(img != img).detach()] = 0
        return img

    @staticmethod
    def swapimdims_3HW_HW3(img):
        """Move the image channels to the first dimension of the numpy
        multi-dimensional array

        :param img: numpy nd array representing the image
        :returns: numpy nd array with permuted axes
        :rtype: numpy nd array

        """
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 1, 2), 0, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 2, 3), 1, 3)

    @staticmethod
    def swapimdims_HW3_3HW(img):
        """Move the image channels to the last dimensiion of the numpy
        multi-dimensional array

        :param img: numpy nd array representing the image
        :returns: numpy nd array with permuted axes
        :rtype: numpy nd array

        """
        if img.ndim == 3:
            return np.swapaxes(np.swapaxes(img, 0, 2), 1, 2)
        elif img.ndim == 4:
            return np.swapaxes(np.swapaxes(img, 1, 3), 2, 3)

    @staticmethod
    def load_image(img_filepath, normaliser, mode, t_size):
        """Loads an image from file as a numpy multi-dimensional array

        :param img_filepath: filepath to the image
        :returns: image as a multi-dimensional numpy array
        :rtype: multi-dimensional numpy array

        """
        myimage = np.array(Image.open(img_filepath))

        if mode == 'True':
            w_input = t_size[0]
            h_input = t_size[1]
            myimage = np.array(cv2.resize(myimage, (w_input, h_input)))

        img = ImageProcessing.normalise_image(myimage, normaliser)  # NB: imread normalises to 0-1
        return img

    @staticmethod
    def normalise_image(img, normaliser):
        """Normalises image data to be a float between 0 and 1

        :param img: Image as a numpy multi-dimensional image array
        :returns: Normalised image as a numpy multi-dimensional image array
        :rtype: Numpy array

        """
        img = img.astype('float32') / normaliser
        return img

    @staticmethod
    def compute_mse(original, result):
        """Computes the mean squared error between to RGB images represented as multi-dimensional numpy arrays.

        :param original: input RGB image as a numpy array
        :param result: target RGB image as a numpy array
        :returns: the mean squared error between the input and target images
        :rtype: float

        """
        return ((original - result) ** 2).mean()

    @staticmethod
    def compute_psnr(image_batchA, image_batchB, max_intensity):
        """Computes the PSNR for a batch of input and output images

        :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param max_intensity: maximum intensity possible in the image (e.g. 255)
        :returns: average PSNR for the batch of images
        :rtype: float

        """
        num_images = image_batchA.shape[0]
        psnr_val = 0.0

        for i in range(0, num_images):
            imageA = image_batchA[i, 0:3, :, :]
            imageB = image_batchB[i, 0:3, :, :]
            imageB = np.maximum(0, np.minimum(imageB, max_intensity))
            psnr_val += 10 * \
                        np.log10(max_intensity ** 2 /
                                 ImageProcessing.compute_mse(imageA, imageB))

        return psnr_val / num_images

    @staticmethod
    def compute_ssim(image_batchA, image_batchB):
        """Computes the SSIM for a batch of input and output images

        :param image_batchA: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param image_batchB: numpy nd-array representing the image batch A of shape Bx3xWxH
        :param max_intensity: maximum intensity possible in the image (e.g. 255)
        :returns: average PSNR for the batch of images
        :rtype: float

        """
        num_images = image_batchA.shape[0]
        ssim_val = 0.0
        for i in range(0, num_images):
            imageA = ImageProcessing.swapimdims_3HW_HW3(
                image_batchA[i, 0:3, :, :])
            imageB = ImageProcessing.swapimdims_3HW_HW3(
                image_batchB[i, 0:3, :, :])
            ssim_val += ssim(imageA, imageB, data_range=imageA.max() - imageA.min(), multichannel=True,
                             gaussian_weights=True, win_size=11)

        return ssim_val / num_images

    @staticmethod
    def hsv_to_rgb(img):
        """Converts a HSV image to RGB
        PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
        Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

        :param img: HSV image
        :returns: RGB image
        :rtype: Tensor

        """
        img = torch.clamp(img, 0, 1)
        img = img.permute(0, 3, 2, 1)

        m1 = 0
        m2 = (img[:, :, :, 2] * (1 - img[:, :, :, 1]) - img[:, :, :, 2]) / 60
        m3 = 0
        m4 = -1 * m2
        m5 = 0

        r = img[:, :, :, 2] + torch.clamp(img[:, :, :, 0] * 360 - 0, 0, 60) * m1 + torch.clamp(
            img[:, :, :, 0] * 360 - 60, 0,
            60) * m2 + torch.clamp(
            img[:, :, :, 0] * 360 - 120, 0, 120) * m3 + torch.clamp(img[:, :, :, 0] * 360 - 240, 0,
                                                                    60) * m4 + torch.clamp(
            img[:, :, :, 0] * 360 - 300, 0, 60) * m5

        m1 = (img[:, :, :, 2] - img[:, :, :, 2] * (1 - img[:, :, :, 1])) / 60
        m2 = 0
        m3 = -1 * m1
        m4 = 0

        g = img[:, :, :, 2] * (1 - img[:, :, :, 1]) + torch.clamp(img[:, :, :, 0] * 360 - 0, 0, 60) * m1 + torch.clamp(
            img[:, :, :, 0] * 360 - 60,
            0, 120) * m2 + torch.clamp(img[:, :, :, 0] * 360 - 180, 0, 60) * m3 + torch.clamp(
            img[:, :, :, 0] * 360 - 240, 0,
            120) * m4

        m1 = 0
        m2 = (img[:, :, :, 2] - img[:, :, :, 2] * (1 - img[:, :, :, 1])) / 60
        m3 = 0
        m4 = -1 * m2

        b = img[:, :, :, 2] * (1 - img[:, :, :, 1]) + torch.clamp(img[:, :, :, 0] * 360 - 0, 0, 120) * m1 + torch.clamp(
            img[:, :, :, 0] * 360 -
            120, 0, 60) * m2 + torch.clamp(img[:, :, :, 0] * 360 - 180, 0, 120) * m3 + torch.clamp(
            img[:, :, :, 0] * 360 - 300, 0, 60) * m4

        img = torch.stack((r, g, b), 3)
        img[(img != img).detach()] = 0

        img = img.permute(0, 3, 2, 1)
        img = img.contiguous()
        img = torch.clamp(img, 0, 1)

        return img

    @staticmethod
    def hsv_to_rgb_new(img, device):
        assert (img.shape[1] == 3)

        h, s, v = img[:, 0]*360, img[:, 1], img[:, 2]
        h_ = (h - torch.floor(h / 360) * 360) / 60
        c = s * v
        x = c * (1 - torch.abs(torch.fmod(h_, 2) - 1))

        zero = torch.zeros_like(c)
        y = torch.stack((
            torch.stack((c, x, zero), dim=1),
            torch.stack((x, c, zero), dim=1),
            torch.stack((zero, c, x), dim=1),
            torch.stack((zero, x, c), dim=1),
            torch.stack((x, zero, c), dim=1),
            torch.stack((c, zero, x), dim=1),
        ), dim=0)

        index = torch.repeat_interleave(torch.floor(h_).unsqueeze(1), 3, dim=1).unsqueeze(0).to(torch.long)
        rgb = (y.gather(dim=0, index=index) + (v - c)).squeeze(0)
        return rgb.to(device)

    @staticmethod
    def rgb_to_hsv(img, device):
        img = img.to(device)
        img = torch.clamp(img, 1e-9, 1)

        img = img.permute(0, 3, 2, 1)
        shape = img.shape

        img = img.contiguous()
        img = img.view(img.shape[0], -1, 3)

        mx = torch.max(img, 2)[0]
        mn = torch.min(img, 2)[0]

        ones = Variable(torch.FloatTensor(
            torch.ones((img.shape[0], img.shape[1])))).to(device)
        zero = Variable(torch.FloatTensor(torch.zeros(shape[0:3]))).to(device)

        img = img.view(shape)

        ones1 = ones[:, 0:math.floor((ones.shape[1] / 2))]
        ones2 = ones[:, math.floor(ones.shape[1] / 2):(ones.shape[1])]

        mx1 = mx[:, 0:math.floor((ones.shape[1] / 2))]
        mx2 = mx[:, math.floor(ones.shape[1] / 2):(ones.shape[1])]
        mn1 = mn[:, 0:math.floor((ones.shape[1] / 2))]
        mn2 = mn[:, math.floor(ones.shape[1] / 2):(ones.shape[1])]

        df1 = torch.add(mx1, torch.mul(ones1 * -1, mn1))
        df2 = torch.add(mx2, torch.mul(ones2 * -1, mn2))

        df = torch.cat((df1, df2), 0)
        del df1, df2
        df = df.view(shape[0:3]) + 1e-10
        mx = mx.view(shape[0:3])

        img = img.to(device)
        df = df.to(device)
        mx = mx.to(device)

        g = img[:, :, :, 1].clone().to(device)
        b = img[:, :, :, 2].clone().to(device)
        r = img[:, :, :, 0].clone().to(device)

        img_copy = img.clone()

        img_copy[:, :, :, 0] = (((g - b) / df) * r.eq(mx).float() + (2.0 + (b - r) / df)
                                * g.eq(mx).float() + (4.0 + (r - g) / df) * b.eq(mx).float())
        img_copy[:, :, :, 0] = img_copy[:, :, :, 0] * 60.0

        zero = zero.to(device)
        img_copy2 = img_copy.clone()

        img_copy2[:, :, :, 0] = img_copy[:, :, :, 0].lt(zero).float(
        ) * (img_copy[:, :, :, 0] + 360) + img_copy[:, :, :, 0].ge(zero).float() * (img_copy[:, :, :, 0])

        img_copy2[:, :, :, 0] = img_copy2[:, :, :, 0] / 360

        del img, r, g, b

        img_copy2[:, :, :, 1] = mx.ne(zero).float() * (df / mx) + \
                                mx.eq(zero).float() * (zero)
        img_copy2[:, :, :, 2] = mx

        img_copy2[(img_copy2 != img_copy2).detach()] = 0

        img = img_copy2.clone()

        img = img.permute(0, 3, 2, 1)
        img = torch.clamp(img, 1e-9, 1)

        return img

    @staticmethod
    def rgb_to_hsv_new(img, device, epsilon=1e-10):
        assert (img.shape[1] == 3)

        r, g, b = img[:, 0], img[:, 1], img[:, 2]
        max_rgb, argmax_rgb = img.max(1)
        min_rgb, argmin_rgb = img.min(1)

        max_min = max_rgb - min_rgb + epsilon

        h1 = 60.0 * (g - r) / max_min + 60.0
        h2 = 60.0 * (b - g) / max_min + 180.0
        h3 = 60.0 * (r - b) / max_min + 300.0

        h = torch.stack((h2, h3, h1), dim=0).gather(dim=0, index=argmin_rgb.unsqueeze(0)).squeeze(0)/360
        s = max_min / (max_rgb + epsilon)
        v = max_rgb

        return torch.stack((h/360, s, v), dim=1).to(device)

    @staticmethod
    def apply_curve(img, C, slope_sqr_diff, channel_in, channel_out, device, clamp=False):
        """Applies a peicewise linear curve defined by a set of knot points to
        an image channel

        :param img: image to be adjusted
        :param C: predicted knot points of curve
        :returns: adjusted image
        :rtype: Tensor

        """
        # img = img.unsqueeze(0)
        # C = C.unsqueeze(0)
        # slope_sqr_diff = slope_sqr_diff.unsqueeze(0)

        curve_steps = C.shape[1] - 1

        '''
        Compute the slope of the line segments
        '''
        slope = C[:, 1:] - C[:, :-1]
        slope_sqr_diff = slope_sqr_diff + torch.sum((slope[:, 1:] - slope[:, :-1]) ** 2, 1)[:, None]

        r = img[:, None, :, :, channel_in].repeat(1, slope.shape[1] - 1, 1, 1) * curve_steps

        s = torch.arange(slope.shape[1] - 1)[None, :, None, None].repeat(img.shape[0], 1, img.shape[1], img.shape[2])

        r = r.to(device)
        s = s.to(device)
        r = r - s

        sl = slope[:, :-1, None, None].repeat(1, 1, img.shape[1], img.shape[2]).to(device)
        scl = torch.mul(sl, r)

        sum_scl = torch.sum(scl, 1) + C[:, 0:1, None].repeat(1, img.shape[1], img.shape[2]).to(device)
        img_copy = img.clone()

        img_copy[:, :, :, channel_out] = img[:, :, :, channel_out] * sum_scl

        img_copy = torch.clamp(img_copy, 0, 1)
        return img_copy, slope_sqr_diff

    @staticmethod
    def adjust_hsv(img, S, device):
        img = img.permute(0, 3, 2, 1)
        shape = img.shape
        img = img.contiguous()

        S1 = torch.exp(S[:, 0:int(S.shape[1] / 4)])
        S2 = torch.exp(S[:, (int(S.shape[1] / 4)):(int(S.shape[1] / 4) * 2)])
        S3 = torch.exp(S[:, (int(S.shape[1] / 4) * 2):(int(S.shape[1] / 4) * 3)])
        S4 = torch.exp(S[:, (int(S.shape[1] / 4) * 3):(int(S.shape[1] / 4) * 4)])

        slope_sqr_diff = Variable(torch.zeros(img.shape[0], 1) * 0.0).to(device)

        '''
        Adjust Hue channel based on Hue using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, S1, slope_sqr_diff, channel_in=0, channel_out=0, device=device)

        '''
        Adjust Saturation channel based on Hue using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S2, slope_sqr_diff, channel_in=0, channel_out=1, device=device)

        '''
        Adjust Saturation channel based on Saturation using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S3, slope_sqr_diff, channel_in=1, channel_out=1, device=device)

        '''
        Adjust Value channel based on Value using the predicted curve
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, S4, slope_sqr_diff, channel_in=2, channel_out=2, device=device)

        img = img_copy.clone()
        del img_copy

        img[(img != img).detach()] = 0

        img = img.permute(0, 3, 2, 1)
        img = img.contiguous()
        return img, slope_sqr_diff

    @staticmethod
    def adjust_rgb(img, R, device):
        img = img.permute(0, 3, 2, 1)
        shape = img.shape
        img = img.contiguous()

        '''
        Extract the parameters of the three curves
        '''
        R1 = torch.exp(R[:, 0:int(R.shape[1] / 3)])
        R2 = torch.exp(R[:, (int(R.shape[1] / 3)):(int(R.shape[1] / 3) * 2)])
        R3 = torch.exp(R[:, (int(R.shape[1] / 3) * 2):(int(R.shape[1] / 3) * 3)])

        '''
        Apply the curve to the R channel
        '''
        slope_sqr_diff = Variable(torch.zeros(img.shape[0], 1) * 0.0).to(device)

        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, R1, slope_sqr_diff, channel_in=0, channel_out=0, device=device)

        '''
        Apply the curve to the G channel
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, R2, slope_sqr_diff, channel_in=1, channel_out=1, device=device)

        '''
        Apply the curve to the B channel
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, R3, slope_sqr_diff, channel_in=2, channel_out=2, device=device)

        img = img_copy.clone()
        del img_copy

        img[(img != img).detach()] = 0

        img = img.permute(0, 3, 2, 1)
        img = img.contiguous()
        return img, slope_sqr_diff

    @staticmethod
    def adjust_lab(img, L, device):
        img = img.permute(0, 3, 2, 1)

        shape = img.shape
        img = img.contiguous()

        '''
        Extract predicted parameters for each L,a,b curve
        '''
        L1 = torch.exp(L[:, 0:int(L.shape[1] / 3)])
        L2 = torch.exp(L[:, (int(L.shape[1] / 3)):(int(L.shape[1] / 3) * 2)])
        L3 = torch.exp(L[:, (int(L.shape[1] / 3) * 2):(int(L.shape[1] / 3) * 3)])

        slope_sqr_diff = Variable(torch.zeros(img.shape[0], 1) * 0.0).to(device)

        '''
        Apply the curve to the L channel
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img, L1, slope_sqr_diff, channel_in=0, channel_out=0, device=device)

        '''
        Now do the same for the a channel
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, L2, slope_sqr_diff, channel_in=1, channel_out=1, device=device)

        '''
        Now do the same for the b channel
        '''
        img_copy, slope_sqr_diff = ImageProcessing.apply_curve(
            img_copy, L3, slope_sqr_diff, channel_in=2, channel_out=2, device=device)

        img = img_copy.clone()
        del img_copy

        img[(img != img).detach()] = 0

        img = img.permute(0, 3, 2, 1)
        img = img.contiguous()

        return img, slope_sqr_diff
