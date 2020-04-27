from datetime import datetime
from scipy.io import loadmat
from pathlib import Path


def calc_age(taken, dob):
    '''
    Oblicza wiek na podstawie daty urodzenia
    :param taken:   data zrobienia zdjęcia
    :param dob:     data urodzenia
    :return:        wiek
    '''

    # dane w kolumnie 'date of birth' są podane jako liczba dni od początku kalendarza gregoriańskego,
    # dlatego wykonujemy konwersję na datetime
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    return taken - birth.year - 1


def load_data(dataset='wiki', data_dir=r'C:\Studia\S8\AgecGAN\DATA\wiki_crop'):
    '''
    Laduje dane z pliku '.mat', przelicza wiek
    :param dataset:     nazwa datasetu
    :param data_dir:    ścieżka do datasetu
    :return:            lista ścieżek do zdjęc, lista lat osób na zdjęciach
    '''

    # ładowanie metadanych
    meta_path = Path(data_dir) / f'{dataset}.mat'
    meta = loadmat(meta_path)
    meta_data = meta[dataset][0, 0]

    # ładowanie ścieżek do zdjęć
    full_path = meta_data['full_path'][0]
    full_path = [y for x in full_path for y in x]

    # ładowanie dat urodzenia
    dob = meta_data['dob'][0]

    # ładowanie daty zrobienia zdjecia
    photo_taken = meta_data['photo_taken'][0]

    # przeliczanie wieku
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    # mapowanie {ścieżka, wiek} dla wieku > 0
    clean_mapping = {pth: age for (pth, age) in zip(full_path, age) if age > 0}

    # podział na listy
    full_path = list(clean_mapping.keys())
    age = list(clean_mapping.values())

    return full_path, age
