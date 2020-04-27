from PIL import Image
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt


import load_data


def scale(x, feature_range=(-1, 1)):
    '''
    Przeskalowanie pikseli obrazu x z zakresu (0,1) do (-1,1)
    :param x:               obraz
    :param feature_range:   docelowy zakres skalowania
    '''

    x_min, x_max = feature_range
    x = x * (x_max - x_min) + x_min
    return x

# granice przedziałów wiekowych
bins = [18, 29, 39, 49, 59]

def one_hot(x, bins):
    '''
    Skalowanie tensora na tensora jednorazowy
    :param x:       tensor
    :param bins:    zakresy przedziałów wiekowych
    '''

    # sprawdzenie, w którym przedziale jest każdy element tensora
    x = x.numpy()
    indices = np.digitize(x, bins, right=True)
    indices = indices.reshape(-1,1)

    # na podstawie indeksu zwraca macierz - wektor wektorów jednorazowych z 1 na pozycji odpowiedniego przedziału
    z = torch.zeros(len(x), len(bins)+1).scatter_(1, torch.tensor(indices), 1)
    return z


class ImageAgeDataset(Dataset):

    '''
    element obraz - wiek osoby na obrazie
    uzyskany na podst. datasetu
    '''

    def __init__(self, dataset, data_dir, transform=None):
        '''
        :param dataset:     nazwa datasetu
        :param data_dir:    ściezka dostępu do datasetu
        :param transform:   opcjonalne przekształcenie
        '''
        self.data_dir = data_dir
        self.full_path, self.age = load_data.load_data(dataset, data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.age)

    def __getitem__(self, idx):
        '''
        zwraca słownik w postaci {obraz, wiek} (otwiera obraz!!!)
        '''
        image = Image.open(os.path.join(self.data_dir, self.full_path[idx]))
        age = self.age[idx]
        sample = {'image': image, 'age': age}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Resize(object):
    '''
    zmiana wymiaru obrazu
    '''

    def __init__(self, output_size):
        '''
        :param output_size: para (wysokość, szerokość), do jakiej zmieniamy rozmiar obrazu
        '''
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        '''
        :param sample:  para {obraz,wiek}
        '''
        image, age = sample['image'], sample['age']

        # gotowa funkcja PyTorcha do zmiany wymiarów obrazu
        image = transforms.Resize(self.output_size)(image)
        return {'image': image, 'age': age}


class ToTensor(object):
    '''
    konwerscja obrazów z tablic np na tensory PyTorcha
    '''

    def __call__(self, sample):

        '''
        :param sample:  para {obraz, wiek}
        '''

        image, age = sample['image'], sample['age']
        image = transforms.ToTensor()(image)

        # jeśli obraz jest w skali szarości - rozszerzamy go na 3 warstwy
        if image.size()[0] == 1:
            image = image.expand(3, -1, -1)

        return {'image': image, 'age': age}



def view_samples(epoch, samples, ages):
    '''
    wyświetlanie kilku obrazów w jednej figurze
    :param epoch:   numer epoki, po której mają być wyświetlone obrazy (dla -1 najnowsza wersja)
    :param samples: lista wyników sampli po każdej z epok uczenia
    :param ages:    lista lat poszczególnych sampli
    '''
    fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
    for ax, img, age in zip(axes.flatten(), samples[epoch], ages):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img +1)*255 / (2)).astype(np.uint8) # rescale to pixel range (0-255)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_title(age)
        im = ax.imshow(img.reshape((64,64,3)))
