"""
Author: Nirmalendu B Patra
"""

import os
import json
import random
import numpy as np
from PIL import Image

class TIGDataset(object):
    def __init__(self, topdir, json_files, how_many_classes=6, seed=101):
        self.topdir = os.path.expanduser(topdir)
        self.json_files = json_files
        self.how_many_classes = how_many_classes
        self.dataset = {k: [] for k in range(how_many_classes)}
        self.duplicates = []
        self._consolidate()
        random.seed(seed)
        for label in self.dataset:
            random.shuffle(self.dataset[label])

    def _consolidate(self):
        already_seen = set()
        for jf, jdir in self.json_files:
            # print(os.path.join(self.topdir, jf))
            with open(os.path.join(self.topdir, jf)) as f:
                jdata = json.loads(f.read())
                for img,label in jdata.items():
                    img_path = os.path.join(self.topdir, jdir, img)
                    if img_path not in already_seen:
                        already_seen.add(img_path)
                        self.dataset[label].append(img_path)
                    else:
                        self.duplicates.append(img_path)

    def samples(self, d_how_many):
        res = {}
        for label, num in d_how_many.items():
            res[label] = []
            for img in self.dataset[label][:num]:
                if os.path.exists(img):
                    res[label].append(self._image(img))
            # res[label] = self.dataset[label][:num]
        return res

    def _image(self, fname):
        img = Image.open(fname)
        img.load()
        return np.asarray(img, dtype="int32" )

    def __repr__(self):
        dup = len(self.duplicates)
        total = 0
        s = "TIGDataset\n"
        for k in self.dataset.keys():
            samples = len(self.dataset[k])
            total += samples
            s += f"  + label {k}: {samples}\n"
        s +=f"  - {dup} duplicates found, total {total} images"
        return s

if __name__ == '__main__':
    pass
