"""Usage: python test_open.py DATASET /path/to/checkpoint
"""

from argparse import ArgumentParser
import os
import glob

from PIL import Image
from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F

from backbones import iresnet50
from config import get_config


parser = ArgumentParser()
parser.add_argument('dataset', choices=['APD', 'APD2', 'APD3'])
parser.add_argument('ckpt')
args = parser.parse_args()

cfg = get_config(args.dataset)
test_root = f"{args.dataset}/test/open"
device = 'cuda:0'
test_batch_size = 256

test_transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5]*3, std=[.5]*3)
])


def load_img(path):
    img = Image.open(path)
    ts = test_transform(img)
    img.close()
    return ts


@torch.no_grad()
def model_forward(batch, backbone):
    features = F.normalize(backbone(batch))
    return features
    
def test_verify(backbone, data):
    preds = []
    batch1 = []
    batch2 = []
    current_batch_size = 0
    for i in range(0, len(data), 2):
        face1 = test_transform(Image.open(data[i]))
        face2 = test_transform(Image.open(data[i+1]))

        if current_batch_size > test_batch_size:
            batch = torch.stack(batch1+batch2)
            features = model_forward(batch.to(device), backbone)
            feat1, feat2 = torch.chunk(features, 2)
            pred = (feat1 * feat2).sum(dim=1).cpu()
            preds.append(pred)
            batch1.clear()
            batch2.clear()
            current_batch_size = 0

        batch1.append(face1)
        batch2.append(face2)
        current_batch_size += 2

    if current_batch_size > 0:
        batch = torch.stack(batch1+batch2)
        features = model_forward(batch.to(device), backbone)
        feat1, feat2 = torch.chunk(features, 2)
        pred = (feat1 * feat2).sum(dim=1).cpu()
        preds.append(pred)

    preds = torch.cat(preds)
    assert len(preds) == (len(data) // 2)

    return preds.numpy()
        

def cmp(name):
    keys = name.split('.')[0].split('_')
    return int(keys[2]), int(keys[3])

    
if __name__ == '__main__':
    backbone = iresnet50(False, dropout=0, fp16=cfg.fp16).to(device)
    backbone_pth = os.path.join(args.ckpt, 'backbone.pth')
    backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(device)))
    backbone.eval()

    data = glob.glob(os.path.join(test_root, '*.jpg'))
    data.sort(key=cmp)
    print("Test data pair:", len(data)//2)
    
    preds = test_verify(backbone, data)
    if not os.path.isdir('result'):
        os.mkdir('result')
    save_path = 'result/' + os.path.basename(args.ckpt)
    np.save(save_path, preds)
    print("Open set result save at", save_path)