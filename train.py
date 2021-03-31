import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from PIL import ImageFile
from configure_training import config
from models.dataset import SRDataset
from models.generator import Generator
from models.discriminator import Discriminator
from models.chopped_vgg19 import ChoppedVGG19
ImageFile.LOAD_TRUNCATED_IMAGES = True

def pretrain(img_data_loader, num_epochs=100, decay_factor=0.1, initial_lr=0.0001, checkpoint=None, save=True):

    if checkpoint is not None:
        imported_checkpoint = torch.load(checkpoint)
        generator = imported_checkpoint['generator']
        starting_epoch = imported_checkpoint['epoch'] + 1
        generator_optimizer = imported_checkpoint['generator_optimizer']
    else:
        generator = Generator()
        generator_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()), lr=initial_lr)
        starting_epoch = 0

    pretrain_criterion = F.mse_loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Push generator to gpu if it's available
    generator.to(device)

    generator.train()
    for epoch in range(starting_epoch, num_epochs):
        # If we're halfway through, reduce learning rate
        if epoch == num_epochs//2:
            for group in generator_optimizer.param_groups:
                group['lr'] = group['lr']*decay_factor

        running_loss = 0.0
        # Iterate through the dataloader
        for ii, (hr_imgs, lr_imgs) in enumerate(tqdm(img_data_loader)):
            hr_imgs, lr_imgs = hr_imgs.to(device), lr_imgs.to(device)

            # Super-resolve low-resolution images
            sr_imgs = generator(lr_imgs)

            # Compute loss, backpropagate, and update generator
            loss = pretrain_criterion(sr_imgs, hr_imgs)
            generator_optimizer.zero_grad()
            loss.backward()
            generator_optimizer.step()
            
            # Increment running loss
            running_loss += loss.item()
            del hr_imgs, lr_imgs, sr_imgs

        print("Pretraining epoch {}, Average loss: {}".format(epoch, running_loss/len(img_data_loader)))

    if save:
        # Save the final pretrained model if you're going to continue later
        torch.save({'epoch': epoch,
                    'generator': generator,
                    'generator_optimizer': generator_optimizer},
                    'pretrained_celebahq_generator.pth.tar')
    del generator

def adversarial_train(img_data_loader, vgg_cutoff_layer=36, num_epochs=2000, decay_factor=0.1, initial_lr=0.0001, adversarial_loss_weight=0.001, checkpoint=None, save=True):
    if checkpoint is not None:
        imported_checkpoint = torch.load(checkpoint)
        generator = imported_checkpoint['generator']
        starting_epoch = 0
        discriminator = Discriminator()
        generator_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()), lr=initial_lr)
        discriminator_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()), lr=initial_lr)
    else:
        generator = Generator()
        starting_epoch = 0
        discriminator = Discriminator()
        generator_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()), lr=initial_lr)
        discriminator_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()), lr=initial_lr)
    
    vgg = ChoppedVGG19(vgg_cutoff_layer)
    
    # generator_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()), lr=initial_lr)
    # discriminator_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()), lr=initial_lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Push everything to gpu if it's available
    content_criterion = nn.MSELoss().to(device)
    adversarial_criterion = nn.BCEWithLogitsLoss().to(device)
    generator.to(device)
    discriminator.to(device)
    vgg.to(device)

    for epoch in range(starting_epoch, num_epochs):
        running_perceptual_loss = 0.0
        running_adversarial_loss = 0.0
        for ii, (hr_imgs, lr_imgs) in enumerate(tqdm(img_data_loader)):
            hr_imgs, lr_imgs = hr_imgs.to(device), lr_imgs.to(device)

            # Forwardpropagate through generator
            sr_imgs = generator(lr_imgs)
            sr_vgg_feature_maps = vgg(sr_imgs)
            hr_vgg_feature_maps = vgg(hr_imgs).detach()

            # Try and discriminate fakes
            sr_discriminator_logprob = discriminator(sr_imgs)
            
            # Calculate loss for generator
            content_loss = content_criterion(sr_vgg_feature_maps, hr_vgg_feature_maps)
            adversarial_loss = adversarial_criterion(sr_discriminator_logprob, torch.ones_like(sr_discriminator_logprob))
            perceptual_loss = content_loss + adversarial_loss_weight*adversarial_loss
            running_perceptual_loss += perceptual_loss.item()
            del sr_vgg_feature_maps, hr_vgg_feature_maps, sr_discriminator_logprob

            # Backpropagate and update generator
            generator_optimizer.zero_grad()
            perceptual_loss.backward()
            generator_optimizer.step()

            # Now for the discriminator
            sr_discriminator_logprob = discriminator(sr_imgs.detach())
            hr_discriminator_logprob = discriminator(hr_imgs)
            adversarial_loss = adversarial_criterion(sr_discriminator_logprob, torch.zeros_like(sr_discriminator_logprob)) + adversarial_criterion(hr_discriminator_logprob, torch.ones_like(hr_discriminator_logprob))
            running_adversarial_loss += adversarial_loss.item()

            # Backpropagate and update discriminator
            discriminator_optimizer.zero_grad()
            adversarial_loss.backward()
            discriminator_optimizer.step()
            del lr_imgs, hr_imgs, sr_imgs, sr_discriminator_logprob, hr_discriminator_logprob
        print("Epoch number {}".format(epoch))
        print("Average Perceptual Loss: {}".format(running_perceptual_loss/len(img_data_loader)))
        print("Average Adversarial Loss: {}".format(running_adversarial_loss/len(img_data_loader)))

        if save:
        # Save the final pretrained model if you're going to continue later
            torch.save({'epoch': epoch,
                        'generator': generator,
                        'generator_optimizer': generator_optimizer,
                        'discriminator': discriminator,
                        'discriminator_optimizer':discriminator_optimizer},
                        'adversarial_training_checkpoint_CelebA_HQ.pth.tar')


if __name__ == '__main__':
    # Import pretrain hyperparameters from config file
    pretrain_epochs = config.HYPERPARAMS.pretrain_epochs
    pretrain_decay_factor = config.HYPERPARAMS.pretrain_decay_factor
    pretrain_initial_learning_rate = config.HYPERPARAMS.pretrain_initial_learning_rate

    # Import adversarial training hyperparameters from config file
    vgg_cutoff_layer = config.HYPERPARAMS.vgg_cutoff_layer
    adversarial_initial_learning_rate = config.HYPERPARAMS.adversarial_initial_learning_rate
    adversarial_epochs = config.HYPERPARAMS.adversarial_epochs
    adversarial_decay_factor = config.HYPERPARAMS.adversarial_decay_factor
    adversarial_loss_weight = config.HYPERPARAMS.adversarial_loss_weight

    # Import dataset info from config file
    data_dir = config.DATASET.img_dir
    pretrain_batch_size = config.DATASET.pretrain_batch_size
    adversarial_batch_size = config.DATASET.adversarial_batch_size
    downsample_factor = config.DATASET.downsample_factor
    crop_size = config.DATASET.crop_size
    shuffle_train = config.DATASET.shuffle_train

    # Create dataset
    train_data = SRDataset(data_dir, 'train', downsample_factor=downsample_factor, crop_size=crop_size)

    # Create trainloader for pretraining and pretrain
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=pretrain_batch_size, shuffle=shuffle_train, num_workers=4)
    pretrain(trainloader)

    # Create trainloader for adversarial training and train
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=adversarial_batch_size, shuffle=shuffle_train, num_workers=4)
    adversarial_train(trainloader, checkpoint='C:/Users/17175/Documents/PyTorch-SRGAN/pretrained_celebahq_generator.pth.tar', num_epochs=1)
