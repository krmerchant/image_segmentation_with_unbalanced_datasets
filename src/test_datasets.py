from datasets import KittiDataset
import matplotlib.pyplot as plt


def main():
    kitti = KittiDataset('dataset.csv', '../data/kitti_semantic/training')

    for(i , sample) in enumerate(kitti):
        if i == 3:
            fig, ax = plt.subplots(1,3)
            ax[0].imshow(sample['image'])
            ax[1].imshow(sample['seg_mask'])
            ax[2].imshow(sample['seg_rgb_mask'])

    plt.show()

if __name__ == "__main__":
    main()