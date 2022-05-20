import argparse
import time
import torch
import yaml
import logging
import numpy as np
import datetime
import os.path
import os
import evaluate
import sys
import model
from dataset import CustomDataset
from utils import prepare_optimizer, prepare_model, prepare_logging, test_gpu_cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

np.set_printoptions(threshold=sys.maxsize)
torch.cuda.empty_cache()


def main(config):
    if config['fix_seed']:
        torch.random.manual_seed(config['fix_seed'])
        torch.manual_seed(config['fix_seed'])

    test_gpu_cuda()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    pretrained_weights = config['pretrained_weights']
    n_workers = config['num_workers']
    fix_seed = config['fix_seed']
    result_path = config['result_path']

    train_path = config['train_settings']['train_path']
    train_batch_size = config['train_settings']['train_batch_size']
    num_epoch = config['train_settings']['num_epochs']
    train_shuffle = str(config['train_settings']['shuffle'])
    target_size = (config['train_settings']['w_input'], config['train_settings']['h_input'])
    random_resize = config['train_settings']['random_resize']
    random_crop = config['train_settings']['random_crop']
    mixed_precision = config['train_settings']['mixed_precision']

    valid_path = str(config['valid_settings']['valid_path'])
    valid_batch_size = config['valid_settings']['valid_batch_Size']
    valid_every = config['valid_settings']['do_every']
    valid_device = str(config['valid_settings']['device'])
    valid_log = config['valid_settings']['log']
    valid_plot = config['valid_settings']['plot']

    test_path = str(config['test_settings']['test_path'])
    test_batch_size = config['test_settings']['test_batch_Size']
    test_every = config['test_settings']['do_every']
    test_device = str(config['test_settings']['device'])
    test_log = config['test_settings']['log']
    test_plot = config['test_settings']['plot']

    opt_name = str(config['optimizer']['name'])
    learning_rate = float(config['optimizer']['lr'])
    weight_decay = float(config['optimizer']['weight_decay'])
    weight_decouple = config['optimizer']['weight_decouple']
    eps = float(config['optimizer']['eps'])
    grad_clip = float(config['optimizer']['grad_clip_norm'])

    warmup = float(config['optimizer']['decay']['warmup'])
    min_lr = float(config['optimizer']['decay']['min_lr'])
    gamma = float(config['optimizer']['decay']['gamma'])

    # making saving directories
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.abspath(os.path.join(result_path, timestamp))
    checkpoints_path = os.path.abspath(os.path.join(save_path, "checkpoints"))

    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    configs = {
        'datetime': timestamp,
        'fix_seed': fix_seed,
        'device': device,
        'workers': n_workers,
        'mixed_precision': mixed_precision,
        'pretrained_weights': pretrained_weights,
        'augment': f"random resize: {random_resize}, random crop: {random_crop}",
        'epoch': num_epoch,
        'batch': train_batch_size,
        'optimizer': f"name: {opt_name}, lr: {learning_rate}, weight decay: {weight_decay}, "
                     f"weight decouple: {weight_decouple}, eps: {eps}, grad_clip_norm: {grad_clip}",
        'lr_scheduler': f"warmup epochs: {warmup}, min lr: {min_lr}, gamma: {gamma}",
        'shuffle': train_shuffle,
        'resolution': target_size,
        'losses': "with regularization",
        'data_details': f'train: {train_path}, valid: {valid_path}, test: {test_path}'
    }

    with open(os.path.join(save_path, "configs.txt"), 'w') as f:
        for k, v in configs.items():
            f.write(str(k) + ': ' + str(v) + '\n\n')

    print('configs:')
    for i, config in configs.items():
        print('\t', '-', i, ":", config)

    prepare_logging(save_path, train_path, valid_path, test_path)

    train_writer = SummaryWriter(os.path.join(save_path, 'train', 'tensorboard_logs'))

    # preparing dataset and dataloader
    training_dataset = CustomDataset(data_dir=train_path, target_size=target_size,
                                     random_resize=random_resize, random_crop=random_crop)
    validation_dataset = CustomDataset(data_dir=valid_path, target_size=target_size,
                                       random_resize=False, random_crop=False)
    test_dataset = CustomDataset(data_dir=test_path, target_size=target_size,
                                 random_resize=False, random_crop=False)

    training_dataloader = DataLoader(training_dataset, batch_size=train_batch_size, shuffle=False,
                                     num_workers=n_workers)
    valid_dataloader = DataLoader(validation_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=n_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=n_workers)

    valid_evaluator = evaluate.Evaluator(valid_dataloader, set_name='valid', save_path=save_path,
                                         device=valid_device, log=valid_log, plot=valid_plot)

    test_evaluator = evaluate.Evaluator(test_dataloader, set_name='test', save_path=save_path,
                                        device=test_device, log=test_log, plot=test_plot)

    net = prepare_model(device, pretrained_weights, print_model=False)
    loss_fn = model.NEW_CURLLoss(ssim_window_size=5, device=device)
    optimizer, lr_scheduler = prepare_optimizer(name=opt_name, net=net, lr=learning_rate, min_lr=min_lr,
                                                train_samples=training_dataset.num_samples, batch=train_batch_size,
                                                epochs=num_epoch, warmup=warmup, gamma=gamma, wd=weight_decay,
                                                weight_decouple=weight_decouple, eps=eps)
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    num_batches = int(np.ceil(training_dataset.num_samples / train_batch_size))

    logging.info(f'fix seed: {fix_seed}')
    logging.info(f'device: {device}')
    logging.info(f'epoch steps : {num_batches}')
    logging.info(f'######### training started with the name of {timestamp} #########')

    for epoch in range(num_epoch):
        running_loss = 0.0
        start = time.time()
        for batch_num, (input_img_batch, gt_img_batch, _) in enumerate(training_dataloader):
            optimizer.zero_grad()
            net.train()
            with torch.cuda.amp.autocast(enabled=mixed_precision):
                input_img_batch = input_img_batch.to(device)
                gt_img_batch = gt_img_batch.to(device)

                pred_img_batch, gradient_regularizer = net(input_img_batch)
                pred_img_batch = torch.clamp(pred_img_batch, 0.0, 1.0)

                loss = loss_fn(pred_img_batch, gt_img_batch, gradient_regularizer)

            if mixed_precision:
                optimizer.zero_grad()
                scaler.scale(loss).backward(retain_graph=True)
                # scaler.scale(rgb_loss_value).backward(retain_graph=True)
                # scaler.scale(cosine_rgb_loss_value).backward(retain_graph=True)
                # scaler.scale(l1_loss_value).backward(retain_graph=True)
                # scaler.scale(hsv_loss_value).backward()
                # scaler.scale(ssim_loss_value).backward()
                # print(scaler._growth_tracker)
                # scaler.scale(gradient_regularizer).backward()
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                scaler.step(optimizer)
                scaler.update()
                train_writer.add_scalar('scaled', scaler.get_scale(), (epoch * num_batches + batch_num))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.data[0]

            if (batch_num + 1) % 10 == 0:
                train_writer.add_scalar('lr', lr_scheduler.get_lr()[0], epoch * num_batches + batch_num)
                train_writer.add_scalar('step_loss', loss.data[0], epoch * num_batches + batch_num)

        end = time.time()
        logging.info(f'epoch %d: {int(end - start)}s - train_loss: %.8f' % (epoch + 1, running_loss / num_batches))

        train_writer.add_scalar('epoch_loss', running_loss / num_batches, epoch + 1)

        # valid set
        if (epoch + 1) % int(valid_every) == 0 and (valid_evaluator.log or valid_evaluator.plot):
            logging.info("evaluating the model on the valid set")

            start = time.time()
            valid_psnr, valid_ssim = valid_evaluator.evaluate(net, epoch)
            end = time.time()

            logging.info(
                f'validation set: {int(end - start)}s - valid_psnr: %.3f valid_ssim: %.3f' % (valid_psnr, valid_ssim))

            snapshot_path = os.path.join(checkpoints_path,
                                         f'val-psnr_{valid_psnr}-ssim_{valid_ssim}-epoch_{epoch + 1}-model.pt')

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': 0,
            }, snapshot_path)
            logging.info(f'saving a checkpoint with name of {snapshot_path}')

        # test set
        if (epoch + 1) % int(test_every) == 0 and (test_evaluator.log or test_evaluator.plot):
            logging.info("evaluating the model on the test set")

            start = time.time()
            test_psnr, test_ssim = test_evaluator.evaluate(net, epoch)
            end = time.time()

            logging.info(f'test set: {int(end - start)}s - test_psnr: %.3f test_ssim: %.3f' % (test_psnr, test_ssim))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the CURL neural network on image pairs")

    parser.add_argument(
        "--config_path", "-c", help="The location of curl config file", default='./config.yaml')

    args = parser.parse_args()
    config_path = args.config_path

    with open(config_path) as file:
        config_file = yaml.full_load(file)
    #     for item, doc in config_file.items():
    #         print(item, ":", doc)

    main(config_file)
