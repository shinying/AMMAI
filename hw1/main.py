import os
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['train', 'closed', 'open'])
    args = parser.parse_args()
    
    if args.mode == 'train':
        os.system("python preprocess.py APD2 retinaface/weights/Resnet50_Final.pth")
        os.system('python train.py APD2')
    elif args.mode == 'closed':
        os.system('python test_closed.py APD2 weights/arcface_retina_closed')
    elif args.mode == 'open':
        os.system('python test_open.py APD2 weights/arcface_retina_open')
        os.system('python score.py results/arcface_retina_open.npy 0.3')