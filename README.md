# CURL
Unofficial implementation of CURL: Neural Curve Layers for Global Image Enhancement paper.
We used many parts of the [official](https://github.com/sjmoran/CURL) repository to create this repo.
Since the official repository has some limitation such as running in batch size of 1, we have changed the code.

This repo has these new features compared to the official one:
1. Supporting different batch sizes (more than 1).
2. Adding random resize and random crop augmentation.
3. Adding cosine learning rate decay with warm up.
4. Adding various optimizers including Adam, AdamW and AdaBelief with decoupled weight decay option.
5. Adding mixed precision option (it might not work properly at this stage)
6. Adding gradients clipping.
7. Adding optional arbitrary test set evaluating.
8. Adding tensorboard option for saving all metrics and images of valid and test sets (optional)
9. Using different devices during training step as well as evaluation step.
10. Adding device option for training and evaluation steps.
11. Much readable and cleaner codes!


## Requirements
Just like the original repository.


## Config File

### General Settings
**pretrained_weights**: Path of pretrained weights. \
**num_workers**: Number of workers for data loader (e.g., 4). \
**fix_seed**: Setting fix seed for training (e.g., 2 or False). \
**result_path**: Path for saving the results including tensorboard, logs and checkpoints.

### Training Settings
**train_path**: The path of training images. \
**train_batch_size**: Batch size of training step. \
**num_epochs**: Number of epochs for training. \
**shuffle**: Shuffling image pairs during training.\
**h_input**: Height of images during training (e.g., 1000). \
**w_input**: Width of images during training (e.g., 1000). \
**random_resize**: Random resize augmentation (True or False). \
**random_crop**: Random cropping augmentation (True or False). \
**mixed_precision**: Training using mixed precision for faster training with less gpu memory footprint (True or False).

### Validation Settings
**valid_path**: The path of validation images. \
**valid_batch_Size**: Batch size of validation step. At this moment, it only supports 1. \
**do_every**: Evaluate every n epochs (e.g., 5) \
**device**: Device name for doing validation step (cuda or cpu) \
**plot**: Saving output images into tensorboard during evaluation (True or False) \
**log**: Saving metrics into tensorboard during evaluation (True or False)

### Test Settings
**test_path**: The path of test images. \
**test_batch_Size**: Batch size of test step. At this moment, it only supports 1. \
**do_every**: Evaluate every n epochs (e.g., 5). \
**device**: Device name for doing test step (cuda or cpu). \
**plot**: Saving output images into tensorboard during evaluation (True or False). \
**log**: Saving metrics into tensorboard during evaluation (True or False). 

### Optimizer
**name**: name of optimizers (Adam, AdamW, AdaBelief). \
**lr**: Learning rate. \
**weight_decouple**: Decoupling weight decay (True or False). \
**weight_decay**: Weight decay (e.g., 0.01). \
**eps**: Epsilon value in optimizers (e.g., 1e-8). \
**grad_clip_norm**: Gradient normalization value (e.g., 5). 

### Learning Rate Decay
**warmup**: Number of epochs for warmup. Can be a float number (e.g., 2.5). \
**min_lr**: Minimum learning rate for learning rate scheduler. \
**gamma**: Gamma vale for cosine weight decay parameter (e.g., 1)