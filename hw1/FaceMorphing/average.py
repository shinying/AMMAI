"""Usage: python /training/images/arranged/in/folders /output/folder
"""
import glob
import os
import sys


inputs = sys.argv[1]
output = sys.argv[2]


persons = next(os.walk(inputs))[1]
imglist = open('combo.txt', 'w')
for i in range(len(persons)-1):
    for j in range(i+1, len(persons)):
        img1 = glob.glob(os.path.join(inputs, persons[i], '*.jpg'))[0]
        img2 = glob.glob(os.path.join(inputs, persons[j], '*.jpg'))[0]
        os.system(f'python code/__init__.py --img1 {img1} --img2 {img2} --output {output} --duration 3 --frame 1')
        print(img1, img2, file=imglist)
