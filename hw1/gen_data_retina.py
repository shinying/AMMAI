import glob
import os
import sys

from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import resize
import numpy as np

from retinaface import Detector


def gen(files, detector, labels_path, resized_list_path):
    labels_file = open(labels_path, 'w')
    resized_list = []

    for file in tqdm(files):
        img = Image.open(file)
        if min(img.size) > 800:
            img = resize(img, size=800)
            img.save(file)
            resized_list.append(file)

        boxes, probs, landmarks = detector(img)
        assert len(boxes) == len(landmarks)
        boxes = np.round(boxes[probs > 0.9]).astype(int)
        landmarks = landmarks[probs > 0.9]

        file = os.path.basename(file)
        print('#', file, file=labels_file)
        for box, landmark in zip(boxes, landmarks):
            x1, y1, x2, y2 = box
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            print(x1, y1, (x2-x1), (y2-y1), file=labels_file, end=' ')

            for x, y in landmark:
                print(f"{x:.3f} {y:.3f} 0.0 ", file=labels_file, end='')
            print('0.0', file=labels_file)

    labels_file.close()
    if len(resized_list) > 0:
        with open(resized_list_path, 'w') as resized_list_file:
            for file in resized_list:
                print(file, file=resized_list_file)


if __name__ == '__main__':
    files = glob.glob(os.path.join(sys.argv[1], '*.jpg'))
    files.sort()
    labels_path = sys.argv[2]
    resized_list_path = sys.argv[3]
    
    detector = Detector(pretrained=sys.argv[4])
    gen(files, detector, labels_path, resized_list_path)

