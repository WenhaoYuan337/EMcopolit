import warnings

warnings.filterwarnings('ignore')
import os, tqdm, cv2
import numpy as np

np.random.seed(0)
from PIL import Image
import albumentations as A
import torch.utils.data as data

import torch, time, datetime, tqdm
from sklearn.metrics import confusion_matrix
from segmentation_models_pytorch import create_model
from segmentation_models_pytorch import losses


class Mydataset(data.Dataset):
    def __init__(self, data, aug=None):
        self.data = data
        self.aug = aug

    def __getitem__(self, index):
        image, label = cv2.imread(f'{self.data[index]}/img.png'), cv2.imread(f'{self.data[index]}/label.png', 0)
        if self.aug is not None:
            aug = self.aug(image=image, mask=label)
            image, label = aug['image'], aug['mask']
        label[label > 0] = 1
        return np.transpose(image, axes=[2, 0, 1]), np.array(label, dtype=np.int_)

    def __len__(self):
        return len(self.data)


def get_data(BATCH_SIZE=32):
    with open('train.txt') as f:
        train_data = list(map(lambda x: x.strip(), f.readlines()))

    with open('test.txt') as f:
        test_data = list(map(lambda x: x.strip(), f.readlines()))

    train_aug = A.Compose([
        A.Resize(512, 512),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.GaussNoise(p=0.5),
        A.Normalize(),
    ])

    test_aug = A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
    ])

    train_dataset = Mydataset(train_data, train_aug)
    train_dataset = torch.utils.data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_dataset = Mydataset(test_data, test_aug)
    test_dataset = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    return train_dataset, test_dataset


def metrice(cm):
    pa = np.diag(cm).sum() / (cm.sum() + 1e-7)

    mpa_arr = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    mpa = np.nanmean(mpa_arr)

    MIoU = np.diag(cm) / np.maximum((np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)), 1)
    MIoU = np.nanmean(MIoU)

    return pa, mpa, MIoU


def cal_cm(y_true, y_pred):
    y_true, y_pred = y_true.to('cpu').detach().numpy(), np.argmax(y_pred.to('cpu').detach().numpy(), axis=1)
    y_true, y_pred = y_true.reshape((-1)), y_pred.reshape((-1))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(2)))
    return cm


if __name__ == '__main__':
    print('===> Loading datasets')
    EPOCH, BATCH_SIZE = 150, 4
    train_dataset, test_dataset = get_data(BATCH_SIZE)

    print('===> Building models')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model('UnetPlusPlus', encoder_name='resnet18', classes=2).to(DEVICE)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=3e-4, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    loss_fl = losses.FocalLoss(mode='multiclass', alpha=0.25)
    loss_jd = losses.JaccardLoss(mode='multiclass')

    with open('train.log', 'w+') as f:
        f.write('epoch,train_loss,test_loss,train_pa,test_pa,train_mpa,test_mpa,train_miou,test_miou')
    best_miou = 0
    print('{} begin train on {}!'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), DEVICE))
    for epoch in range(EPOCH):
        model.to(DEVICE)
        model.train()
        train_loss, train_cm = 0, np.zeros(shape=(2, 2))
        begin = time.time()
        for x, y in tqdm.tqdm(train_dataset, desc='Epoch {}/{} train stage'.format(epoch + 1, EPOCH)):
            x, y = x.to(DEVICE), y.to(DEVICE).long()
            pred = model(x.float())
            l = loss_fl(pred, y) + loss_jd(pred, y)

            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += float(l.data)
            train_cm += cal_cm(y, pred)
        train_loss /= len(train_dataset)
        train_pa, train_mpa, train_miou = metrice(train_cm)

        val_loss, val_cm = 0, np.zeros(shape=(2, 2))
        model.eval()
        with torch.no_grad():
            for x, y in tqdm.tqdm(test_dataset, desc='Epoch {}/{} val stage'.format(epoch + 1, EPOCH)):
                x, y = x.to(DEVICE), y.to(DEVICE).long()
                pred = model(x.float())
                l = loss_fl(pred, y) + loss_jd(pred, y)
                val_loss += float(l.data)
                val_cm += cal_cm(y, pred)
        val_loss /= len(test_dataset)
        val_pa, val_mpa, val_miou = metrice(val_cm)

        if val_miou > best_miou:
            best_miou = val_miou
            model.to('cpu')
            torch.save(model, 'model.pt')
        print(
            '{} epoch:{}, time:{:.2f}s, lr:{:.6f}, train_loss:{:.4f}, val_loss:{:.4f}, train_pa:{:.4f}, val_pa:{:.4f}, train_mpa:{:.4f}, val_mpa:{:.4f}, train_miou:{:.4f}, val_miou:{:.4f}'.format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch + 1, time.time() - begin, optimizer.state_dict()['param_groups'][0]['lr'], train_loss, val_loss,
                train_pa, val_pa, train_mpa, val_mpa,
                train_miou,
                val_miou
            ))
        with open('train.log', 'a+') as f:
            f.write('\n{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}'.format(
                epoch, train_loss, val_loss, train_pa, val_pa, train_mpa, val_mpa, train_miou, val_miou
            ))
        lr_scheduler.step()
