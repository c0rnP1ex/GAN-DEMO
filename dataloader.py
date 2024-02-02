import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from utils import read_mnist_images, read_mnist_labels

class MNISTDataset(Dataset):
    def __init__(self, img_path, label_path, flag='train'):
        self.images = read_mnist_images(img_path).astype('float32') / 255
        self.labels = read_mnist_labels(label_path)
        
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.images)
        

if __name__ == '__main__':
    dataset = MNISTDataset('data/train-images-idx3-ubyte.gz', 'data/train-labels-idx1-ubyte.gz')
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # get 0 item
    print(dataset[0])
