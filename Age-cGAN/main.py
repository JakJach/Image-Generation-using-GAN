import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

import image_treatment
import GAN


dataset='wiki'
data_dir='C:\Studia\S8\Age-cGAN\DATA\wiki_crop'
bins = [18, 29, 39, 49, 59]
img_size = 64
batch_size = 128
#num_workers = 0

tfms = transforms.Compose([image_treatment.Resize((img_size, img_size)),
                           image_treatment.ToTensor()])

train_dataset = image_treatment.ImageAgeDataset(dataset, data_dir, transform=tfms)

# build DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)

iter(train_loader).next()

plt.imshow(train_dataset[0]['image'].numpy().transpose(1,2,0))

plt.imshow(train_dataset[609]['image'].numpy().transpose(1,2,0))

# obtain one batch of training images
dataiter = iter(train_loader)
data = dataiter.next()
images, labels = data['image'], data['age']

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
plot_size=20

for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.transpose(images[idx], (1, 2, 0)))
    # print out the correct label for each image
    # .item() gets the value contained in a Tensor
    ax.set_title(str(labels[idx].item()))


# define hyperparams
conv_dim = 64
z_size = 100
y_size = 6 # no. of age classes

# define discriminator and generator
D = GAN.Discriminator(y_size, conv_dim)
G = GAN.Generator(z_size, y_size, conv_dim)

print(D)
print('---')
print(G)

# params
lr = 0.0002
beta1=0.5
beta2=0.999 # default value

# Create optimizers for the discriminator and generator
d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])



#%%time

root_dir = r'C:\Studia\S8\Age-cGAN\results'
model = 'GAN_1'
os.makedirs(root_dir, exist_ok=True)

# move models to GPU, if available
device = GAN.device
G.to(device)
D.to(device)

import pickle as pkl

# training hyperparams
num_epochs = 10

# keep track of loss and generated, "fake" samples
samples = []
losses = []

print_every = 300

# Get some fixed data for sampling. These are images that are held
# constant throughout training, and allow us to inspect the model's performance
sample_size=16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()
fixed_y = np.random.randint(len(bins), size=sample_size)
fixed_y = fixed_y.reshape(-1,1)
fixed_y = torch.zeros(sample_size, len(bins)+1).scatter_(1, torch.LongTensor(fixed_y), 1)

# train the network
for epoch in range(num_epochs):

    for batch_i, batch in enumerate(train_loader):

        batch_size = batch['image'].size(0)

        # important rescaling image step
        real_images = image_treatment.scale(batch['image'])

        # one-hot age
        ages = image_treatment.one_hot(batch['age'], bins)

        # ============================================
        #            TRAIN THE DISCRIMINATOR
        # ============================================

        d_optimizer.zero_grad()

        # 1. Train with real images

        # Compute the discriminator losses on real images
        real_images = real_images.to(device)
        ages = ages.to(device)

        D_real = D(real_images, ages)
        d_real_loss = GAN.real_loss(D_real)

        # 2. Train with fake images

        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        # move x to GPU, if available
        z = z.to(device)
        # if train_on_gpu:
        #    z = z.cuda()
        fake_images = G(z, ages)

        # Compute the discriminator losses on fake images
        D_fake = D(fake_images, ages)
        d_fake_loss = GAN.fake_loss(D_fake)

        # add up loss and perform backprop
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # =========================================
        #            TRAIN THE GENERATOR
        # =========================================
        g_optimizer.zero_grad()

        # 1. Train with fake images and flipped labels

        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        z = z.to(device)
        fake_images = G(z, ages)

        # Compute the discriminator losses on fake images
        # using flipped labels!
        D_fake = D(fake_images, ages)
        g_loss = GAN.real_loss(D_fake)  # use real loss to flip labels

        # perform backprop
        g_loss.backward()
        g_optimizer.step()

        # Print some loss stats
        if batch_i % print_every == 0:
            # append discriminator loss and generator loss
            losses.append((d_loss.item(), g_loss.item()))
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                epoch + 1, num_epochs, d_loss.item(), g_loss.item()))

    ## AFTER EACH EPOCH##
    # generate and save sample, fake images
    G.eval()  # for generating samples
    fixed_z = fixed_z.to(device)
    fixed_y = fixed_y.to(device)
    samples_z = G(fixed_z, fixed_y)
    samples.append(samples_z)
    G.train()  # back to training mode

    # Save checkpoint
    GAN.checkpoint(G, D, epoch, model, root_dir)

# Save training generator samples
GAN.save_samples_ages(samples, fixed_y, model, root_dir)


fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()


fixed_y_ages = GAN.oh_to_class(fixed_y)
_ = image_treatment.view_samples(-1, samples, fixed_y_ages)
