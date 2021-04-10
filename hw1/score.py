"""Usage python score.py PREDICTION THRESHOLD
"""
import numpy as np
import sys

gt_file = 'data/test/open_set/labels.txt'
res_file = sys.argv[1]
thresh = float(sys.argv[2])

gt = np.loadtxt(gt_file).astype(bool)
res = np.load(res_file)

score = ((res > thresh).astype(bool) == gt).astype(float).mean()
print(f'Open set acc: {score:.1%}')

