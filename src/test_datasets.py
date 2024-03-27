from datasets import KittiDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from PIL import Image
import numpy as np
import torch


def main():
    kitti = KittiDataset('dataset.csv', '../data/kitti_semantic/training', transforms.Compose([
        transforms.Resize((256, 256))
    ]))
   # kitti = KittiDataset('dataset.csv', '../data/kitti_semantic/training', transforms.Compose([
   #      transforms.CenterCrop(350)
   #  ]))

    for (i, (image, seg)) in enumerate(kitti):
        print(i)
        print(image.shape)
        print(seg.shape)
        if i == 100:
            numpy_image = image.permute(1, 2, 0).numpy()
            numpy_seg = torch.argmax(seg, dim=3)
            numpy_seg = numpy_seg.reshape(seg.shape[1], seg.shape[2]).numpy()
            print(numpy_seg.shape)
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow((numpy_image))
            seg_map = ax[1].imshow(numpy_seg)
            fig.colorbar(seg_map)
            print(numpy_seg)

    data_loader = DataLoader(kitti, batch_size=32, shuffle=True)
    for batch in data_loader:
        print("Got a Batch")
        inputs, labels = batch  # Assuming each batch returns inputs and labels
        print(inputs.shape)
        print(labels.shape)
       # Your training/evaluation loop
    print("Sucessfully interated through batches!")
    plt.show()


if __name__ == "__main__":
    main()
