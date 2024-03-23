from datasets import KittiDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from PIL import Image
def main():
   # kitti = KittiDataset('dataset.csv', '../data/kitti_semantic/training', transforms.Compose([
   #     transforms.Resize((1200,1200))
   # ]))
    kitti = KittiDataset('dataset.csv', '../data/kitti_semantic/training', transforms.Compose([
         transforms.CenterCrop(200)
     ]))
    im = 0; 
    
    for(i ,( image, seg)) in enumerate(kitti):
        print(i)
        print(image.shape)
        print(seg.shape)
        if i==10:
            fig,ax = plt.subplots(1,2)
            numpy_image = image.permute(1,2,0).numpy()
            numpy_seg = seg.numpy()
            ax[0].imshow((numpy_image))
            ax[1].imshow(numpy_seg)


   
    data_loader = DataLoader(kitti, batch_size=32, shuffle=False)
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