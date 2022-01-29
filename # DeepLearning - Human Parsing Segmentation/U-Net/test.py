import torch
import matplotlib.pyplot as plt
from UNET import UNET
from PIL import Image
import torchvision.transforms as transform
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNET(in_channels= 3, out_channels= 1).to(device = DEVICE)

# load model
state = torch.load("./model.pth.tar")
model.load_state_dict(state["state_dict"])

# make sample test img
tf = transform.Compose([transform.ToTensor(), transform.Resize((224, 224))])
tf2 = transform.Compose([transform.ToTensor(), transform.Resize((224, 224))])
image = np.array(Image.open("BUSI2/Sample/test.png").convert("RGB"))
image = tf(image)
image = image.unsqueeze(0).to(device = DEVICE)

# make prediction with test img
prediction = model(image)
prediction = prediction.squeeze(0)
prediction = prediction.squeeze(0)
prediction = torch.sigmoid(prediction)

# threshold
prediction[prediction > 0.5 ] = 1
prediction[prediction <= 0.5] = 0

im = Image.open("BUSI2/Sample/test.png")
im2 = Image.open("BUSI2/Sample/test_mask.png")
# visualize

from UNET import segmap

prediction = prediction.cpu().detach().numpy()
prediction = segmap(prediction)
plt.subplot(1,3,1)
plt.imshow(prediction)
plt.subplot(1,3,2)
plt.imshow(im)
plt.subplot(1,3,3)
plt.imshow(im2)
plt.show()
