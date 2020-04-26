from datetime import datetime
from scipy.io import loadmat
from pathlib import Path


def calc_age(taken, dob):
    '''
    Calculate age
    :param taken: Date when photo taken
    :param dob: Date of birth in serials
    :return: age in years
    '''
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def load_data(dataset='wiki', data_dir='C:\Studia\S8\Kwiecień\GAN_project\DATA\wiki_crop'):
    '''
    Load meta data and calculate age
    :param dataset: dataset name, defaults = 'wiki'
    :param data_dir: data directory, defaults = 'C:\Studia\S8\Kwiecień\GAN_project\DATA\wiki_crop'
    :return: list of full_path and age
    '''
    # Load meta data
    meta_path = Path(data_dir) / f'{dataset}.mat'
    meta = loadmat(meta_path)
    meta_data = meta[dataset][0, 0]

    # Load all file paths
    full_path = meta_data['full_path'][0]
    full_path = [y for x in full_path for y in x]

    # Load dates of birth
    dob = meta_data['dob'][0]

    # Load years when photo taken
    photo_taken = meta_data['photo_taken'][0]

    # Calculate age
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    # Clean mapping with age > 0
    clean_mapping = {pth: age for (pth, age) in zip(full_path, age) if age > 0}

    # List of full_path, age
    full_path = list(clean_mapping.keys())
    age = list(clean_mapping.values())

    return full_path, age
