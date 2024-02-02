import torch
import torch.nn as nn
from dataloader import MNISTDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from models import Generator, Discriminator
import torch.optim as optim
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = MNISTDataset('data/train-images-idx3-ubyte.gz', 'data/train-labels-idx1-ubyte.gz')
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
loss_function = nn.BCELoss()

epochs = 50


for epoch in range(epochs):
    for batch_idx, (real_images, _) in enumerate(train_loader):
        real_images = real_images.to(device)
        real_targets = torch.ones(real_images.size(0), 1, device=device)
        fake_targets = torch.zeros(real_images.size(0), 1, device=device)

        # 训练判别器
        discriminator.zero_grad()
        
        real_scores = discriminator(real_images)
        real_loss = loss_function(real_scores, real_targets)

        noise = torch.randn(real_images.size(0), 100, device=device)
        fake_images = generator(noise)
        fake_scores = discriminator(fake_images.detach())
        fake_loss = loss_function(fake_scores, fake_targets)

        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        # 训练生成器
        generator.zero_grad()
        fake_scores = discriminator(fake_images)
        g_loss = loss_function(fake_scores, real_targets)
        g_loss.backward()
        g_optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
    # save ckpt
    torch.save(generator.state_dict(), f'generator.pth')
    torch.save(discriminator.state_dict(), f'discriminator.pth')

    # generate images
    noise = torch.randn(16, 100).to(device)
    fake_images = generator(noise).cpu().detach()
    fake_images = fake_images.view(-1, 28, 28)
    fake_images = fake_images.numpy()*255
    # save as png
    for i in range(16):
        plt.imsave(f'images/{epoch+1}_{i}.png', fake_images[i], cmap='gray')
    
