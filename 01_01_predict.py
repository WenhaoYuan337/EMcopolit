import os, glob, cv2, shutil
from PIL import Image
import numpy as np
import nibabel as nib
import matplotlib.pylab as plt
np.random.seed(0)

import albumentations as A
import torch, tqdm
import numpy as nps

if __name__ == '__main__':
    colors = [(0, 0, 0), (0, 0, 255)]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_aug = A.Compose([
        A.Resize(512, 512),
        A.Normalize(),
    ])

    # model path
    model = torch.load('model.pt').to(DEVICE)
    model.eval()
    
    path = 'your data'
    save_path = '/result'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    for i in os.listdir(path):
        ori_image = cv2.imread(f'{path}/{i}')
        image = test_aug(image=ori_image)['image']
        image = np.expand_dims(np.transpose(image, axes=[2, 0, 1]), axis=0)
        image = torch.from_numpy(image).to(DEVICE).float()
        output = model(image).cpu().detach().numpy()[0].argmax(0)
        ratio = np.sum(output[output == 1]) / (output.shape[0] * output.shape[1])
        ori_image = cv2.imread(f'{path}/{i}')
        image = test_aug(image=ori_image)['image']
        image = np.expand_dims(np.transpose(image, axes=[2, 0, 1]), axis=0)
        image = torch.from_numpy(image).to(DEVICE).float()
        output = model(image).cpu().detach().numpy()[0].argmax(0)
        output = np.reshape(np.array(colors, np.uint8)[np.reshape(output, [-1])], [512, 512, -1])
        mask = cv2.resize(output, (ori_image.shape[1], ori_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        output = cv2.addWeighted(ori_image, 0.5, mask, 0.5, 0)
        filename_output = os.path.join(save_path, str(i))
        filename_mask = os.path.join(save_path, 'mask' + str(i))
        cv2.imwrite(filename_output, output)
        cv2.imwrite(filename_mask, mask)
