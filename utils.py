import gzip
import numpy as np

def read_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        # 跳过前16个字节的文件头信息
        f.read(16)
        # 读取剩余数据，每张图像为28x28像素
        data = np.frombuffer(f.read(), dtype=np.uint8)
    # 调整数据形状为[图像数量, 高度, 宽度]
    return data.reshape(-1, 28, 28)

def read_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        # 跳过前8个字节的文件头信息
        f.read(8)
        # 读取剩余数据，每个标签为1字节
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels