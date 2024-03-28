from models import UNet
import unittest
from datasets import KittiDataset
from torchvision import transforms, utils


class TestUNetNetwork(unittest.TestCase):

    def test_construction(self):
        unet = UNet(50)

    def test_pass_sample(self):
        kitti = KittiDataset('dataset.csv', '../data/kitti_semantic/training', transforms.Compose([
            transforms.Resize((1024, 1024))
        ]))
        [images, seg] = kitti[0]
        unet = UNet(37)
        output = unet(images)
        c,w,h = output.shape
        self.assertEqual([37,1024, 1024], [c,w,h])


if __name__ == '__main__':
    unittest.main()
