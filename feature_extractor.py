import torch
from model import Model
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import utils

memory_data = utils.EMNISTPair(root='data', train=True, transform=utils.test_transform, download=True,split="letters")
memory_loader = DataLoader(memory_data, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)


# 2. Specify the path to the saved model state dictionary
save_name_pre = "128_0.5_200_256_50"
# Replace with the actual name you used
model_path = '{}_modelEMNISTPair.pth'.format(save_name_pre)
# 3. Load the saved state dictionary
model=torch.load(model_path)

# 4. Load the state dictionary into the model
# model.load_state_dict(state_dict)


total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_loader, desc='Feature extracting'):
            feature, out = model(data.cuda(non_blocking=True))
            print(out)
            feature_bank.append(out)

        # [D, N]
        feature_bank_X = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_loader.dataset.targets, device=feature_bank_X.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(memory_loader)



feature_bank_X = torch.cat(feature_bank, dim=0).contiguous()


type(feature_bank_X)

import torch
import pandas as pd

df = pd.DataFrame(feature_bank_X.cpu().numpy(), columns=[f'col_{i}' for i in range(128)])


df.shape

feat=feature_labels.cpu().numpy()

feat.shape

feat = feat.reshape(-1, 1)


feat.shape

df.insert(0, 'feat', feat)

df.shape


df.to_csv('Emnist_resnet_cc_128.csv', index=False)

print("Excel file created: output.xlsx")


