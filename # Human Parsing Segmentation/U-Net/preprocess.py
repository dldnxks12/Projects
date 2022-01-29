import os
import shutil

# make directory for renamed image
#os.mkdir("./train")
#os.mkdir("./train-mask")

# read file names in source folder
datafiles = os.listdir("/-Project1-7. Segmentation-Semantic/U-Net/BUSI2/data/")
maskfiles = os.listdir("/-Project1-7. Segmentation-Semantic/U-Net/BUSI2/mask/")

# from
datasrc = "C:/Users/USER/PycharmProjects/A.I/-Project1-7. Segmentation-Semantic/U-Net/BUSI2/data/"
masksrc = "C:/Users/USER/PycharmProjects/A.I/-Project1-7. Segmentation-Semantic/U-Net/BUSI2/mask/"

# to
datadst = "C:/Users/USER/PycharmProjects/A.I/-Project1-7. Segmentation-Semantic/U-Net/BUSI2/train/"
maskdst = "C:/Users/USER/PycharmProjects/A.I/-Project1-7. Segmentation-Semantic/U-Net/BUSI2/train-mask/"


# image rename and move to new folder
for idx, file in enumerate(datafiles):
    shutil.move(datasrc + file , datadst + str(idx) +".png")

for idx, file in enumerate(maskfiles):
    shutil.move(masksrc + file , maskdst + str(idx) +".png")





