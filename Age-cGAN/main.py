import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

# nasze paczki
import image_treatment
import GAN


dataset = 'wiki'
data_dir = 'C:\Studia\S8\Age-cGAN\DATA\wiki_crop'

# granice przedziałów wiekowych
bins = [18, 29, 39, 49, 59]

# rozmiar obrazka
img_size = 64

# wielkość paczki
batch_size = 128

# kombinacja zmieniania rozmiaru obrazu i konwerscji do tensora
tfms = transforms.Compose([image_treatment.Resize((img_size, img_size)),
                           image_treatment.ToTensor()])

# załadowanie datasetu i transformacja
train_dataset = image_treatment.ImageAgeDataset(dataset, data_dir, transform=tfms)

# użycie loadera PyTorcha
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)
iter(train_loader).next()

# wyświetlenie kilku przykładowych obrazków w celu sprawdzenia
plt.imshow(train_dataset[21]['image'].numpy().transpose(1,2,0))
plt.imshow(train_dataset[37]['image'].numpy().transpose(1,2,0))
plt.imshow(train_dataset[420]['image'].numpy().transpose(1,2,0))
plt.imshow(train_dataset[609]['image'].numpy().transpose(1,2,0))

# pobranie paczki danych z datasetu
dataiter = iter(train_loader)
data = dataiter.next()
images, labels = data['image'], data['age']

# wyświetlenie obrazków z pobranego batcha
fig = plt.figure(figsize=(25, 4))
plot_size=20

for idx in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(str(labels[idx].item()))


# parametry obu sieci
conv_dim = 64
z_size = 100
y_size = 6

# stworzenie obu sieci
D = GAN.Discriminator(y_size, conv_dim)
G = GAN.Generator(z_size, y_size, conv_dim)

print(D)
print('---')
print(G)

# parametry optymalizatorów - oba stworzone dla takich samych parametrów
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

# stworzenie optymalizatorów
d_optimizer = optim.Adam(D.parameters(), lr, [beta1, beta2])
g_optimizer = optim.Adam(G.parameters(), lr, [beta1, beta2])


#%%time

# nazwa modelu
model = 'GAN_1'
root_dir = r'C:\Studia\S8\Age-cGAN\results'
os.makedirs(root_dir, exist_ok=True)

# sprawdzenie możliwości wykonania na GPU
device = GAN.device
G.to(device)
D.to(device)

# parametry uczenia
num_epochs = 10
print_every = 300

# lista, w której zapisane zostaną wygenerowane "fałszywe" obrazy
samples = []

# lista wartości strat procesu uczenia
losses = []

# przyszykowanie wygenerowanych danych do kontrolowania procesu
sample_size=16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
fixed_z = torch.from_numpy(fixed_z).float()
fixed_y = np.random.randint(len(bins), size=sample_size)
fixed_y = fixed_y.reshape(-1,1)
fixed_y = torch.zeros(sample_size, len(bins)+1).scatter_(1, torch.LongTensor(fixed_y), 1)

# proces uczenia
for epoch in range(num_epochs):

    for batch_i, batch in enumerate(train_loader):

        batch_size = batch['image'].size(0)

        # skalowanie wszystkich obrazów z paczki
        real_images = image_treatment.scale(batch['image'])

        # wyznaczenie wektorów jednostkowych dla paczki
        ages = image_treatment.one_hot(batch['age'], bins)

        # ============================================
        #               DYSKRYMINATOR
        # ============================================

        d_optimizer.zero_grad()

        # I. Trening na prawdziwych obrazkach

        # obliczenie strat dla uczenia przy prawdziwych obrazkach
        real_images = real_images.to(device)
        ages = ages.to(device)
        D_real = D(real_images, ages)
        d_real_loss = GAN.real_loss(D_real)

        # II. Trening na fałszywych obrazkach

        # generacja (używamy genratora!)
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        z = z.to(device)
        fake_images = G(z, ages)

        # obliczenie strat dla uczenia przy fałszywych obrazkach
        D_fake = D(fake_images, ages)
        d_fake_loss = GAN.fake_loss(D_fake)

        # dodanie strat i propagacja wsteczna
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # =========================================
        #               GENERATOR
        # =========================================
        g_optimizer.zero_grad()

        # I. Trening z fałszywymi obrazami i odwróconymi labelami

        # generacja szumu
        z = np.random.uniform(-1, 1, size=(batch_size, z_size))
        z = torch.from_numpy(z).float()
        z = z.to(device)
        fake_images = G(z, ages)

        # obliczenie strat dyskryminatora dla fałszywych obrazków z odwróconymi labelami
        D_fake = D(fake_images, ages)
        g_loss = GAN.real_loss(D_fake)

        # propagacja wsteczna
        g_loss.backward()
        g_optimizer.step()

        # wyświetlenie wyników (strat) tej epoki
        if batch_i % print_every == 0:
            # dodanie strat do listy
            losses.append((d_loss.item(), g_loss.item()))

            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                epoch + 1, num_epochs, d_loss.item(), g_loss.item()))

    # Po każdej epoce generujemy nowe sample i zapisujemy je w liście

    # generacja
    G.eval()  # for generating samples
    fixed_z = fixed_z.to(device)
    fixed_y = fixed_y.to(device)
    samples_z = G(fixed_z, fixed_y)
    samples.append(samples_z)
    # trening generatora
    G.train()

    GAN.checkpoint(G, D, epoch, model, root_dir)

# zapis wygenerowanych sampli
GAN.save_samples_ages(samples, fixed_y, model, root_dir)

# wykres wyników sieci
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
plt.plot(losses.T[1], label='Generator', alpha=0.5)
plt.title("Training Losses")
plt.legend()

# wyświetlenie wygenerowanych sampli
fixed_y_ages = GAN.oh_to_class(fixed_y)
_ = image_treatment.view_samples(-1, samples, fixed_y_ages)
