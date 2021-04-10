"""Usage: python preprocess.py path/to/output/root/dir path/to/face/detector/checkpoint
"""

import glob
import os
import sys

from PIL import Image
from torchvision.transforms.functional import resize
from tqdm import tqdm

from cropalign import FaceDetector


detector = FaceDetector(device='cuda:0', pretrained=sys.argv[2])


def imread(file, max_size=800):
    img = Image.open(file)
    if min(img.size) > 800:
        img = resize(img, size=max_size)
    return img


def process_train(files, output):
    multifaces = []
    n_faces = 0
    names = set()

    for f in tqdm(files):
        img = imread(f)
        aligned_img, is_multi = detector(img, keep_all=False, select='max')
        
        if aligned_img is None:
            multifaces.append(f+' > no face')
            print(f, 'does not contain images')
            img.close()
            continue

        if is_multi:
            multifaces.append(f)
        else:
            base = os.path.basename(f)
            name = base.split('.')[0].split('_')[0]
            if name not in names:
                os.mkdir(os.path.join(output, name))
                names.add(name)
            Image.fromarray(aligned_img).save(os.path.join(output, name, base))
            n_faces += 1
        img.close()

    with open('multifaces_train.txt', 'w') as f:
        for p in multifaces:
            print(p, file=f)

    print(f"{n_faces} faces and {len(names)} people are detected")
    print(len(multifaces), "are filtered")


def process_test(files, output):
    multifaces = []
    for i in tqdm(range(0, len(files), 2)):
        img1 = Image.open(files[i])
        aligned_img1, is_multi1 = detector(img1)
        img2 = Image.open(files[i+1])
        aligned_img2, is_multi2 = detector(img2)

        if is_multi1 or is_multi2:
            multifaces.append(files[i])
            multifaces.append(files[i+1])

        Image.fromarray(aligned_img1).save(os.path.join(output, os.path.basename(files[i])))
        Image.fromarray(aligned_img2).save(os.path.join(output, os.path.basename(files[i+1])))

    with open('multifaces_test.txt', 'w') as f:
        for p in multifaces:
            print(p, file=f)


def process_test_multi(files, output):
    for i, f in (enumerate(tqdm(files))):
        img = imread(f)
        align_imgs, _ = detector(img, keep_all=True)
        if align_imgs is None:
            print(f, 'does not contain images')
            img.close()
            continue
            
        save_path = os.path.join(output, os.path.basename(f).split('.')[0])
        os.mkdir(save_path)
        for i, aimg in enumerate(align_imgs):
            Image.fromarray(aimg).save(os.path.join(save_path, f"{i}.jpg"))
        img.close()


def process_test_open(files, output):
    for f in tqdm(files):
        img = imread(f)
        align_img, _ = detector(img, keep_all=False, select='center')
        if align_img is None:
            img.close()
            print(f, 'does not contain images')
            continue
        
        save_path = os.path.join(output, os.path.basename(f))
        Image.fromarray(align_img).save(save_path)
        img.close()


def get_files(root):
    files = glob.glob(os.path.join(root, '*.jpg'))
    files.sort(key=lambda f: os.path.basename(f))
    print("Total", len(files), "images")

    return files


if __name__ == '__main__':
    outroot = sys.argv[1]

    print("Generate clean training face crops (filtering images with multi-people)")
    root = "data/train/A"
    output = outroot + "/train/"
    if not os.path.isdir(output):
        os.makedirs(output)

    files = get_files(root)
    process_train(files, output)

    print("\nGenerate test closed set face crops (crop all faces from an image)")
    root = "data/test/closed_set/test_pairs"
    output = outroot + "/test/closed_multi"
    if not os.path.isdir(output):
        os.makedirs(output)

    files = get_files(root)
    process_test_multi(files, output)

    print("\nGenerate test open set face crops (crop the face most close to the image center)")
    root = "data/test/open_set/test_pairs"
    output = outroot + "/test/open"
    if not os.path.isdir(output):
        os.makedirs(output)

    files = get_files(root)
    process_test_open(files, output)
