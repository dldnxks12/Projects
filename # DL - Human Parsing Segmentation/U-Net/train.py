import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CarvanaDataset
from UNET import UNET
from tqdm import tqdm
from tensorboardX import SummaryWriter

# hyperparameters

LEARNING_RATE = 1e-6
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 20
NUM_EPOCHS = 30
TRAIN_IMG_DIR = "/-Project1-7. Segmentation-Semantic/U-Net/BUSI2/train/"
TRAIN_MASK_DIR = "/-Project1-7. Segmentation-Semantic/U-Net/BUSI2/train-mask/"
VAL_IMG_DIR = "/-Project1-7. Segmentation-Semantic/U-Net/BUSI2/test/"
VAL_MASK_DIR = "/-Project1-7. Segmentation-Semantic/U-Net/BUSI2/test-mask/"


def main():

    model = UNET(in_channels= 3, out_channels= 1).to(device = DEVICE)
    loss_fn = nn.BCEWithLogitsLoss() # UNET Class의 마지막 layer에서 sigmoid 해준다면 그냥 nn.BCELoss 쓰고, nn.BCEWithLogitsLoss는 내부적으로 Sigmoid 수행해 줌
    '''
        out_channels가 1개 이상이면 loss_fn = Cross Entropy Loss 사용
    '''
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_dataset = CarvanaDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TFtype = 1)
    train_dataset2 = CarvanaDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TFtype = 2)
    train_dataset3 = CarvanaDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, TFtype = 3)

    train_loader =  DataLoader(dataset = train_dataset, batch_size= BATCH_SIZE, shuffle= True, drop_last= True)
    train_loader2 =  DataLoader(dataset = train_dataset2, batch_size= BATCH_SIZE, shuffle= True, drop_last= True)
    train_loader3 =  DataLoader(dataset = train_dataset3, batch_size= BATCH_SIZE, shuffle= True, drop_last= True)

    val_dataset = CarvanaDataset(VAL_IMG_DIR, VAL_MASK_DIR, TFtype = 1)
    val_loader =  DataLoader(dataset = val_dataset, batch_size= 3, shuffle=True, drop_last=True)

    T_loader = []
    T_loader.append(train_loader)
    T_loader.append(train_loader2)
    T_loader.append(train_loader3)

    writer = SummaryWriter(logdir = "scalar/UNET")

    model.train()
    step = 0

    for epoch in range(NUM_EPOCHS):
        losses = []  # total loss
        accuracies = []  # total acc
        for loader_idx, loader in enumerate(T_loader):
            print(f" Loader {loader_idx} Epoch {epoch}")
            num_correct = 0  # correct pixels
            num_pixels = 0   # number of pixels
            loop = tqdm(loader) # leave = True
            for idx, (data, targets) in enumerate(loop):
                data = data.to(device=DEVICE)
                targets = targets.float().to(device=DEVICE)  # channel dimension 처리

                predictions = model(data)
                loss = loss_fn(predictions, targets)
                losses.append(loss.item())

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Calculate ACC
                predictions = (predictions > 0.5).float()  # 0.5 이상인 Pixel == True or 1
                num_correct += (predictions == targets).sum()  # True == 1 , False == 0
                num_pixels += torch.numel(predictions)

                running_train_acc = float(num_correct) / float(num_pixels)
                accuracies.append(running_train_acc)

                # Write on Tensorboard
                writer.add_scalar('Training_loss', loss, global_step=step)
                writer.add_scalar('Training_ACC', running_train_acc, global_step=step)
                step += 1

                # Write on tqdm
                loop.set_description(f"Epoch[{epoch}/{NUM_EPOCHS}]")
                loop.set_postfix(loss = loss.item(), acc = running_train_acc)

        print(f"total epoch loss {sum(losses) / len(losses)} total epoch acc {sum(accuracies)/len(accuracies)}")

    writer.close()

    print("#--------- train finished ------ #")

    print("save Model")
    state = {
        "state_dict" : model.state_dict(),
        "optimizer" : optimizer.state_dict()
    }

    torch.save(state, "model.pth.tar")

    print("#--------- Valid Start ------ #")
    check_acc(val_loader, model)
    print("#--------- Valid finished ------ #")

def check_acc(val_loader, model, device = 'cude'):

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        print("length of Val_loader : ", len(val_loader))
        for x, y in val_loader:
            x = x.to(device = DEVICE)
            y = y.to(device = DEVICE)

            preds = torch.sigmoid(model(x)) # get hypothesis and do activation function

            preds = ( preds > 0.5 ).float() # 0.5 이상인 Pixel == True or 1

            num_correct += (preds == y).sum() # True == 1 , False == 0
            num_pixels  += torch.numel(preds) # torch.numel : number of element

            dice_score += (2 * (preds * y).sum()) / ( (preds + y).sum() + 1e-8 ) # IoU Score

    print(f" Got {num_correct} / {num_pixels} with ACC {num_correct/num_pixels}")
    print(f"Dice Score : {dice_score/len(val_loader)}")


if __name__ == "__main__":
    main()



