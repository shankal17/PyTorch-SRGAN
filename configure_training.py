from easydict import EasyDict

config = EasyDict()
config.HYPERPARAMS = EasyDict()

# Pretraining hyperparameters
config.HYPERPARAMS.pretrain_epochs = 100
config.HYPERPARAMS.pretrain_decay_factor = 0.1
config.HYPERPARAMS.pretrain_initial_learning_rate = 0.0001

# Adversarial training hyperparamters
config.HYPERPARAMS.vgg_cutoff_layer = 36
config.HYPERPARAMS.adversarial_initial_learning_rate = 0.0001
config.HYPERPARAMS.adversarial_epochs = 2000
config.HYPERPARAMS.adversarial_decay_factor = 0.1
config.HYPERPARAMS.adversarial_loss_weight = 0.001

# Dataset Info
config.DATASET = EasyDict()
config.DATASET.img_dir = 'C:/Users/17175/Documents/CelebA-HQ/train1024x1024'
config.DATASET.pretrain_batch_size = 16
config.DATASET.adversarial_batch_size = 8
config.DATASET.downsample_factor = 4
config.DATASET.crop_size = 96
config.DATASET.shuffle_train = True
