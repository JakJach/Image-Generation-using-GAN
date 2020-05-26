from GAN import Generator, device
from image_treatment import view_samples, one_hot
import torch
import argparse
import numpy as np
WEIGHTS_PATH = "path_to_weights"

CONV_DIM = 64
Z_SIZE = 100
Y_SIZE = 6
BINS = [18, 29, 39, 49, 59]


def main():

    def get_args():
        parser = argparse.ArgumentParser(
            description='Add necessary arguments to run generator model')
        parser.add_argument('path',  type=str,
                            help='path to model weights')

        parser.add_argument('image_size',  type=int,
                            help='number of created images')

        parser.add_argument('age', type=int, help='desired age')

        return parser.parse_args()

    args = get_args()
    path = args.path
    age = args.age
    image_size = args.image_size

    generator = Generator(Z_SIZE, Y_SIZE, CONV_DIM)
    generator.to(device)
    generator.load_state_dict(torch.load(path))
    generator.eval()

    z = np.random.uniform(-1, 1, size=(image_size, Z_SIZE))
    z = torch.from_numpy(z).float()
    z = z.to(device)    
    age_one_hot = one_hot(torch.tensor([age]*image_size), BINS).to(device)
    result = generator(z, age_one_hot)
    view_samples(-1, [result], [age]*image_size)


if __name__ == "__main__":
    main()
