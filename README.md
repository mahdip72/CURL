# CURL
Unofficial pytorch implementation of [CURL: Neural Curve Layers for Global Image Enhancement](https://arxiv.org/abs/1911.13175)
paper.
I used many parts of the [official](https://github.com/sjmoran/CURL) repository to create this repo.
Since the official repository has some limitation such as running in batch size of 1, I have greatly changed the 
original implementation in order to make it more usable.

This repo has these new features compared to the official one:
1. Supporting different batch sizes (more than 1).
2. Adding random resize and random crop augmentation.
3. Adding cosine learning rate decay with warm up.
4. Adding more optimizers. Besides Adam, it includes AdamW and AdaBelief with decoupled weight decay option.
5. Adding mixed precision option (it might not work properly at this stage)
6. Adding gradients clipping as a regularization option.
7. Adding test set evaluation as an optional evaluation step.
8. Adding new metrics such as PCQI.
9. Adding tensorboard option for saving all metrics and images of valid and test sets (optional).
10. Adding separate device option during training and evaluation steps. 
11. Much readable and cleaner codes!


## Requirements
Original requirements + [AdaBelief](https://github.com/juntang-zhuang/Adabelief-Optimizer) +
[torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)

## Config File
Using config.yaml file you can change many settings without going into the codes. In fact, 90 percent of the time you 
only need to change its values and start training. In the following, every option has been explained. 

#### General Settings
**pretrained_weights**: Path of pretrained weights. \
**num_workers**: Number of workers for data loader (e.g., 4). \
**fix_seed**: Setting fix seed for training (e.g., 2 or False). \
**result_path**: Path for saving the results including tensorboard, logs and checkpoints.

#### Training Settings
**train_path**: The path of training images. \
**train_batch_size**: Batch size of training step. \
**num_epochs**: Number of epochs for training. \
**shuffle**: Shuffling image pairs during training.\
**h_input**: Height of images during training (e.g., 1000). \
**w_input**: Width of images during training (e.g., 1000). \
**random_resize**: Random resize augmentation (True or False). \
**random_crop**: Random cropping augmentation (True or False). \
**mixed_precision**: Training using mixed precision for faster training with less gpu memory footprint (True or False).

#### Validation Settings
**valid_path**: The path of validation images. \
**valid_batch_Size**: Batch size of validation step. At this moment, it only supports 1. \
**do_every**: Evaluate every n epochs (e.g., 5) \
**device**: Device name for doing validation step (cuda or cpu) \
**plot**: Saving output images into tensorboard during evaluation (True or False) \
**log**: Saving metrics into tensorboard during evaluation (True or False)

#### Test Settings
**test_path**: The path of test images. \
**test_batch_Size**: Batch size of test step. At this moment, it only supports 1. \
**do_every**: Evaluate every n epochs (e.g., 5). \
**device**: Device name for doing test step (cuda or cpu). \
**plot**: Saving output images into tensorboard during evaluation (True or False). \
**log**: Saving metrics into tensorboard during evaluation (True or False). 

#### Optimizer
**name**: name of optimizers (Adam, AdamW, AdaBelief). \
**lr**: Learning rate. \
**weight_decouple**: Decoupling weight decay (True or False). \
**weight_decay**: Weight decay (e.g., 0.01). \
**eps**: Epsilon value in optimizers (e.g., 1e-8). \
**grad_clip_norm**: Gradient normalization value (e.g., 5). 
 
#### Learning Rate Decay
**warmup**: Number of epochs for warmup. Can be a float number (e.g., 2.5). \
**min_lr**: Minimum learning rate for learning rate scheduler. \
**gamma**: Gamma vale for cosine weight decay parameter (e.g., 1)
 
## Train
Set the config.yaml values as you want and run `python train.py -c [config file path]` 

## Tips for Training Using This Repository
* The code has been tested on a custom dataset and in general, it trains slightly faster than the original implementation.
* AdaBelief optimizer decreases the loss function faster than AdamW.
* Despite, I have refactored and optimized many parts of the codes such as loss functions and data loaders,
it needs more optimization to work efficiently as it should. Mainly, during training,
when it goes for calculating the loss functions, the performance drops significantly.
* Mixed precision option seems to have a problem with MS-SSIM loss function. 
In other words, the loss value changes to NaN during training. I plan to replace the MS-SSIM function with the official
implementation of MS-SSIM in the Pytorch Metric library and test mixed precision option.
* If you set the log and plot options in valid or test set to False in the config file, it is ignored during evaluation.
Need to point out that during valid evaluation, it saves a model checkpoint in the result path.
* Since the original resolution of images in valid and test sets are considered, the gpu can run out of memory in higher
resolution. Therefore, I have provided device option for them in the config file to switch between devices during
evaluation steps.
* In order to train on Adobe5k dataset you need to check the load_adobe_5k_data function in dataset.py.
I borrowed it from the official repository. I have not tested it yet.
But, in general, you need to create a list of pairs in the form of 
`[(input_path, target_dir), (input_path, target_dir), ...]`
as the output of that function to work properly in the code.
* 