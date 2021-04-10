"""Usage: python test_closed.py DATASET /path/to/checkpoint
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
from partial_fc import PartialFC
import losses


parser = ArgumentParser()
parser.add_argument('dataset', choices=['APD', 'APD2', 'APD3'])
parser.add_argument('ckpt')
args = parser.parse_args()

cfg = get_config(args.dataset)
test_root = f"{args.dataset}/test/closed_multi"
ground_truth = "data/test/closed_set/labels.txt"
loss_fn = 'ArcFace'
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
def model_forward(batch, backbone, fc):
    features = F.normalize(backbone(batch.to(device)))
    preds = fc.predict(features)

    return preds


def test_intersec(backbone, fc, datadirs):
    preds1 = []
    preds2 = []
    current_batch_size = 0
    batch1 = []
    batch2 = []
    chunks1 = []
    chunks2 = []
    for i in range(0, len(datadirs), 2):
        data1, data2 = datadirs[i], datadirs[i+1]
        assert '_'.join(data1.split('_')[:-1]) == '_'.join(data2.split('_')[:-1])

        imgs1 = [load_img(path) for path in glob.glob(os.path.join(test_root, data1, '*.jpg'))]
        imgs2 = [load_img(path) for path in glob.glob(os.path.join(test_root, data2, '*.jpg'))]

        if current_batch_size + len(imgs1) + len(imgs2) > test_batch_size:
            batch = torch.cat([torch.stack(batch1), torch.stack(batch2)])
            output = model_forward(batch, backbone, fc)
            res1, res2 = torch.split(output, [len(batch1), len(batch2)])
            res1 = torch.split(res1, chunks1)
            res2 = torch.split(res2, chunks2)
            preds1.extend(res1)
            preds2.extend(res2)

            current_batch_size = 0
            batch1.clear()
            batch2.clear()
            chunks1.clear()
            chunks2.clear()

        batch1.extend(imgs1)
        batch2.extend(imgs2)
        chunks1.append(len(imgs1))
        chunks2.append(len(imgs2))
        current_batch_size += len(imgs1) + len(imgs2)

    if current_batch_size > 0:
        batch = torch.cat([torch.stack(batch1), torch.stack(batch2)])
        output = model_forward(batch, backbone, fc)
        res1, res2 = torch.split(output, [len(batch1), len(batch2)])
        res1 = torch.split(res1, chunks1)
        res2 = torch.split(res2, chunks2)
        preds1.extend(res1)
        preds2.extend(res2)

    assert len(preds1) == len(preds2)
    preds = [len(set(pred1.tolist()).intersection(set(pred2.tolist()))) > 0 \
        for pred1, pred2 in zip(preds1, preds2)]
    gt = open(ground_truth).read().splitlines()
    assert len(preds) == len(gt)

    return (np.array(preds) == np.array(gt).astype(bool)).sum() / len(preds)


def cmp(name):
    keys = name.split('_')
    return int(keys[2]), int(keys[3])


if __name__ == '__main__':
    backbone = iresnet50(False, dropout=0, fp16=cfg.fp16).to(device)
    backbone_pth = os.path.join(args.ckpt, 'backbone.pth')
    backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(device)))
    backbone.eval()

    margin_softmax = eval("losses.{}".format(loss_fn))()
    fc = PartialFC(
        rank=0, local_rank=0, world_size=1, resume=True,
        batch_size=test_batch_size, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
        sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=args.ckpt)
    fc.eval()

    datadirs = next(os.walk(test_root))[1]
    datadirs.sort(key=cmp)
    print("Test data pair:", len(datadirs)//2)

    acc = test_intersec(backbone, fc, datadirs)
    print(f"Closed set test acc: {acc:.1%}")