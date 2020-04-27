import os
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F


# jeśli dostępna jest karta graficzna, to ona wykona obliczenia, jeśli nie, CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Funkcje pomocnicze


def conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
    '''
    tworzy warstwę sieci konwolucyjnej i dodaje do kontenera sekwencyjnego
    :param in_channels:     wejścia warstwy
    :param out_channels:    wyjścia warstwy
    :param kernel_size:     rozmiar kernela
    :param stride:          krok konwolucji
    :param padding:         wypełnienie zerami wejścia
    :param batch_norm:      normalizacja partii
    :return:                kontener sekwencyjny z dodaną warstwą
    '''

    layers = []
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(conv_layer)

    # normalizacja partii
    if batch_norm:
        bn = nn.BatchNorm2d(out_channels)
        layers.append(bn)

    # dodaje stworzoną warstwę od kontenera sekwencyjnego
    return nn.Sequential(*layers)


def conv_trans(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
    '''
    tworzy transponowaną warstwę sieci konwolucyjnej i dodaje do kontenera sekwencyjnego
    :param in_channels:     wejścia warstwy
    :param out_channels:    wyjścia warstwy
    :param kernel_size:     rozmiar kernela
    :param stride:          krok konwolucji
    :param padding:         wypełnienie zerami wejścia
    :param batch_norm:      normalizacja partii
    :return:                kontener sekwencyjny z dodaną warstwą
    '''

    layers = []
    t_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    layers.append(t_conv)

    # normalizacja partii
    if batch_norm:
        bn = nn.BatchNorm2d(out_channels)
        layers.append(bn)

    # dodaje stworzoną warstwę od kontenera sekwencyjnego
    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self, y_size=6, conv_dim=64):
        '''
        inicjazilator dyskryminatora [5 warstw]
        :param conv_dim:    rozmiar pierwszej warstwy (dla naszych obrazów 64x64 mamy 64)
        :param y_size:      długość wektora warunków (dla nas 6 kategorii wiekowych)
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
        propagacja sieci ("przepuszcza" obraz przez sieć)
        :param x:   obraz
        :param y:   wektor warunków
        :result:    wektor jednorazowy (jedynkowy???) z kategorią wiekową
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
        inicjalizator generatora [5 warstw]
        :param z_size:      długość wektora wejścia (szumu)
        :param conv_dim:    szerokość, jaką ma mieć ostatnia warstwa generatora (u nas 64 przez zdjęcia 64x64)
        :param y_size:      długość wektora warunków (dla nas 6 kategorii wiekowych)
        '''

        super(Generator, self).__init__()

        self.conv_dim = conv_dim

        self.t_conv1 = conv_trans(z_size + y_size, conv_dim * 8, 4, 1, 0)
        self.t_conv2 = conv_trans(conv_dim * 8, conv_dim * 4, 4)
        self.t_conv3 = conv_trans(conv_dim * 4, conv_dim * 2, 4)
        self.t_conv4 = conv_trans(conv_dim * 2, conv_dim, 4)
        self.t_conv5 = conv_trans(conv_dim, 3, 4, batch_norm=False)

    def forward(self, z, y):
        '''
        propagacja sieci ("przepuszacza" szum przez sieć i generuje obraz)
        :param x:   szum
        :param y:   wektor warunków
        :return:    wygenerowany obraz (64x64x3)s
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


def real_loss(D_out, smooth=False):
    '''
    oblicza straty sieci w fazie podawania "prawdziwych" obrazów
    :param D_out:   wyjście sieci
    :param smooth:  ustala, czy wygładzić wyniki, czy nie
    :return:        straty sieci
    '''

    batch_size = D_out.size(0)

    # ewentualne wygładzanie współczynnikiem 0.9
    if smooth:
        labels = torch.ones(batch_size)*0.9
    else:
        labels = torch.ones(batch_size)

    # przerzut tensora jedynek na kartę graficzną (jeśli możliwy)
    labels = labels.to(device)

    # obliczanie straty
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    '''
    oblicza straty sieci w fazie wykrywania "fałszywek"
    :param D_out:   wyjście sieci
    :return:        straty sieci
    '''

    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)

    # przerzut tensora zer na kartę graficzną (jeśli możliwy)
    labels = labels.to(device)
    criterion = nn.BCEWithLogitsLoss()

    # obliczanie strat
    loss = criterion(D_out.squeeze(), labels)
    return loss


def checkpoint(G, D, epoch, model, root_dir):
    '''
    awaryjny zapis modelu
    :param G:           generator
    :param D:           dyskryminator
    :param epoch:       numer epoki, po której wykouje się zapis
    :param model:       nazwa modelu
    :param root_dir:    ścieżka dostępu do projektu
    '''

    target_dir = f'{root_dir}/{model}'
    os.makedirs(target_dir, exist_ok=True)

    G_path = os.path.join(target_dir, f'G_{epoch}.pkl')
    D_path = os.path.join(target_dir, f'D_{epoch}.pkl')

    torch.save(G.state_dict(), G_path)
    torch.save(D.state_dict(), D_path)


def oh_to_class(fixed_y):
    '''
    konwertuje wektory jednostkowe (jednorazowy???) do przedziałów wiekowych
    :param fixed_y: lista wektorów jednostkowych
    :return:        lista przedziałów wiekowych
    '''

    # słownik grup wiekowych
    age_map = {0: '0-18', 1: '19-29', 2: '30-39', 3: '40-49', 4: '50-59', 5: '60+'}

    if torch.cuda.is_available():
        fixed_y = fixed_y.cpu()

    fixed_y_idxs = fixed_y.numpy().nonzero()[1]
    fixed_y_ages = [age_map[idx] for idx in fixed_y_idxs]

    return fixed_y_ages


def save_samples_ages(samples, fixed_y, model, root_dir):
    '''
    zapisuje wiek sampli
    :param samples:     sample do zapisania
    :param fixed_y:     lista wektorów jednostkowych
    :param model:       nazwa modelu
    :param root_dir:    ścieżka dostępu do projektu
    :return:
    '''

    fixed_y_ages = oh_to_class(fixed_y)
    samples_ages = {'samples': samples, 'ages': fixed_y_ages}
    target_dir = f'{root_dir}/{model}'
    os.makedirs(target_dir, exist_ok=True)
    with open(f'{target_dir}/train_samples_ages.pkl', 'wb') as f:
        pkl.dump(samples_ages, f)
