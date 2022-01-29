import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transform
import torchvision.transforms.functional as TF
import numpy as np

class CarvanaDataset(Dataset):

    def __init__(self, image_dir, mask_dir, TFtype): # TFtype으로 augumentation 수행
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform.Compose([transform.ToTensor(),
                                            transform.Resize((160, 160))])
        self.TFtype = TFtype  # augumentation type
        self.images = os.listdir(image_dir) # image_dir 경로에 있는 모든 file의 이름을 가져온다. (path X file name O)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])  # Rename as "path + file name" !
        mask_path = os.path.join(self.mask_dir, self.images[idx])  # Rename as "path + file name" !

        # RGB Image로 열어서 numpy type으로 type casting
        image = np.array(Image.open(img_path).convert("RGB")) # 확실히 하기 위해 RGB Image로 열도록 명시
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)  # L : gray Scale

        image2 = image*1.03
        image2 = image2.astype(np.uint8)
        image2 = self.transform(image2) # Image with noise

        image3 = image*0.97
        image3 = image3.astype(np.uint8)
        image3 = self.transform(image3) # Image with noise

        image = self.transform(image)
        mask = self.transform(mask)

        # ---- data augumentation ---- #
        # No augumentation
        if self.TFtype == 1:
            image = image
            mask = mask
        # Horizontal Flip
        elif self.TFtype == 2:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        # Vertical Flip
        elif self.TFtype == 3:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        # noise data
        elif self.TFtype == 4:
            image = image2
            mask = mask
        # noise data + Horizontal Flip
        elif self.TFtype == 5:
            image = TF.hflip(image2)
            mask = TF.hflip(mask)
        # noise data + Vertical Flip
        elif self.TFtype == 6:
            image = TF.vflip(image2)
            mask = TF.vflip(mask)
        # noise2 data
        elif self.TFtype == 7:
            image = image3
            mask = mask
        # noise data2 + Horizontal Flip
        elif self.TFtype == 8:
            image = TF.hflip(image3)
            mask = TF.hflip(mask)
        # noise data2 + Vertical Flip
        elif self.TFtype == 9:
            image = TF.vflip(image3)
            mask = TF.vflip(mask)


        '''
        elif self.TFtype == 10: # Horizontal + Vertical Flip
            image = TF.hflip(image)
            image = TF.vflip(image)
            mask = TF.hflip(mask)
            mask = TF.vflip(mask)
                    
        elif self.TFtype == 11: # Horizontal + Vertical Flip + Noise
            image = TF.hflip(image2)
            image = TF.vflip(image)
            mask = TF.hflip(mask)
            mask = TF.vflip(mask)                                
        '''

        # 0.0 ~ 255.0
        mask[mask == 255.0] = 1.0 # Sigmoid 를 마지막 Activation function으로 사용할 것이기 때문에 1의 값으로 맞추어주자.

        return image, mask


