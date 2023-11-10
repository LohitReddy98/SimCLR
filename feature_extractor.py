import torch
from model import Model
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import utils

memory_data = utils.MNISTPair(root='data', train=True, transform=utils.test_transform, download=True)
memory_loader = DataLoader(memory_data, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

model_path = 'models/epoch10_512_mnist.pth'
model=torch.load(model_path)

total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_loader, desc='Feature extracting'):
            feature, out = model(data.cuda(non_blocking=True))
            feature_bank.append(out)

        # [D, N]
        feature_bank_X = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_loader.dataset.targets, device=feature_bank_X.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(memory_loader)



feature_bank_X = torch.cat(feature_bank, dim=0).contiguous()


import torch
import pandas as pd

df = pd.DataFrame(feature_bank_X.cpu().numpy(), columns=[f'col_{i}' for i in range(512)])


print(df.shape)
feat=feature_labels.cpu().numpy()

print(feat.shape)

feat = feat.reshape(-1, 1)


feat.shape

df.insert(0, 'feat', feat)

print(df.shape)


df.to_csv('mnist_resnet_cc_256.csv', index=False)

print("Excel file created: output.xlsx")
