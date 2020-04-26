import os
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F


# Helpers
def conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(conv_layer)

    if batch_norm:
        bn = nn.BatchNorm2d(out_channels)
        layers.append(bn)

    return nn.Sequential(*layers)


# Helpers
def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
    layers = []
    t_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(t_conv)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self, y_size, conv_dim=64):
        '''
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        :param y_size: The number of conditions
        '''
        super(Discriminator, self).__init__()
        self.conv_dim = conv_dim
        self.y_size = y_size
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)
        self.conv2 = conv(conv_dim + y_size, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)
        self.conv5 = conv(conv_dim * 8, 1, 4, 1, 0, batch_norm=False)

    def forward(self, x, y):
        '''
        Forward propagation of the neural network
        :param x: The input scaled image x
        :param y: One-hot encoding condition tensor y (N,y_size)
        :return: Discriminator logits; the output of the neural network
        '''
        x = F.relu(self.conv1(x))
        y = y.view(-1, y.size()[-1], 1, 1)
        y = y.expand(-1, -1, x.size()[-2], x.size()[-1])
        x = torch.cat([x, y], 1)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)

        return x


class Generator(nn.Module):

    def __init__(self, z_size, y_size, conv_dim=64):
        '''
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        '''
        super(Generator, self).__init__()

        self.conv_dim = conv_dim

        # self.fc = nn.Linear(z_size+y_size, conv_dim*8*4*4)
        self.t_conv1 = deconv(z_size + y_size, conv_dim * 8, 4, 1, 0)
        self.t_conv2 = deconv(conv_dim * 8, conv_dim * 4, 4)
        self.t_conv3 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.t_conv4 = deconv(conv_dim * 2, conv_dim, 4)
        self.t_conv5 = deconv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, z, y):
        '''
        Forward propagation of the neural network
        :param x: The input to the neural network
        :param y: The input condition to the neural network, Tensor (N,y_size)
        :return: A 64x64x3 Tensor image as output
        '''
        x = torch.cat([z, y], dim=1)
        x = x.view(-1, x.size()[-1], 1, 1)
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
        x = self.t_conv5(x)
        x = torch.tanh(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size) # real labels = 1
    # move labels to GPU if available
    labels = labels.to(device)
    # binary cross entropy with logits loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size) # fake labels = 0
    labels = labels.to(device)
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss



def checkpoint(G, D, epoch, model, root_dir):
    target_dir = f'{root_dir}/{model}'
    os.makedirs(target_dir, exist_ok=True)
    G_path = os.path.join(target_dir, f'G_{epoch}.pkl')
    D_path = os.path.join(target_dir, f'D_{epoch}.pkl')
    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)


def oh_to_class(fixed_y):
    age_map = {0: '0-18', 1: '19-29', 2: '30-39', 3: '40-49', 4: '50-59', 5: '60+'}
    if torch.cuda.is_available():
        fixed_y = fixed_y.cpu()
    fixed_y_idxs = fixed_y.numpy().nonzero()[1]
    fixed_y_ages = [age_map[idx] for idx in fixed_y_idxs]

    return fixed_y_ages


def save_samples_ages(samples, fixed_y, model, root_dir):
    fixed_y_ages = oh_to_class(fixed_y)
    samples_ages = {'samples': samples, 'ages': fixed_y_ages}
    target_dir = f'{root_dir}/{model}'
    os.makedirs(target_dir, exist_ok=True)
    with open(f'{target_dir}/train_samples_ages.pkl', 'wb') as f:
        pkl.dump(samples_ages, f)
