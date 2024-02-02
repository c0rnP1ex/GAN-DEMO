import torch
from models import Generator
import matplotlib.pyplot as plt
generator = Generator().to('cuda')
generator.load_state_dict(torch.load('generator.pth'))
noise = torch.randn(1600, 100).to('cuda')
fake_images = generator(noise).cpu().detach()
fake_images = fake_images.view(-1, 28, 28)
fake_images = fake_images.numpy()*255
for i in range(1600):
    plt.imsave(f'images_gen/{i}.png', fake_images[i], cmap='gray')
