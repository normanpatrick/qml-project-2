"""
Author: Nirmalendu B Patra
Copyright (C) Nirmalendu B Patra - All Rights Reserved
"""

import os
import json
import random
import numpy as np
from PIL import Image
import skimage.transform

class TIGDataAccess(object):
    """
    - Pre-process, save normalized, non-normalized all in npy format
    - Should behave like a seamless cache
    """
    def __init__(self,
                 dir_in,
                 json_files,
                 how_many_classes,
                 dir_preprocess):
        self.dir_in = dir_in
        self.dir_preprocess = dir_preprocess
        self.json_files = json_files
        self.how_many_classes=how_many_classes
        self.td = None

    """
    - pre-process all images
    - save normalized, non-normalized all in npy format
    - allow for various resolutions
    - API should support sklearn.train_test_split
    """
    def setup(self,
              resize,
              normalize,
              max_items_per_label=0):
        if self.td is None:
            self.td = TIGDataset(topdir=self.dir_in,
                                 json_files=self.json_files,
                                 how_many_classes=self.how_many_classes,
                                 resize=resize,
                                 normalize=normalize,
                                 max_items_per_label=max_items_per_label)
            total = 0
            for label in range(self.how_many_classes):
                for img in self.td.next_sample(label=label):
                    total += 1
            print(f"Total processed: {total}")

class TIGDataset(object):
    def __init__(self,
                 topdir,
                 json_files,
                 how_many_classes,
                 max_items_per_label=0,
                 resize=None,
                 normalize=True):
        self.topdir = os.path.expanduser(topdir)
        self.json_files = json_files
        self.how_many_classes = how_many_classes
        self.dataset = {k: [] for k in range(how_many_classes)}
        self.duplicates = []
        self.resize = resize
        self.normalize = normalize
        # if n > 0, then process only n items, otherwise all
        self.max_items_per_label = max_items_per_label
        self._consolidate()
        # random.seed(seed)
        # for label in self.dataset:
        #     random.shuffle(self.dataset[label])

    def _consolidate(self):
        already_seen = set()
        totals = {k:0 for k in range(self.how_many_classes)}
        for jf, jdir in self.json_files:
            # print(os.path.join(self.topdir, jf))
            with open(os.path.join(self.topdir, jf)) as f:
                jdata = json.loads(f.read())
                for img,label in jdata.items():
                    if self.max_items_per_label == totals[label] \
                       and self.max_items_per_label > 0:
                        continue
                    img_path = os.path.join(self.topdir, jdir, img)
                    if img_path not in already_seen:
                        totals[label] += 1
                        already_seen.add(img_path)
                        self.dataset[label].append(img_path)
                    else:
                        self.duplicates.append(img_path)

    def next_sample(self, label):
        for item in self.dataset[label]:
            yield item

    def _read_single_image(self, img):
        assert os.path.exists(img) == True
        imdata = self._image(img)
        if self.normalize:
            imdata = self._image(img) / 255 # np.linalg.norm(imdata)
        return imdata

    def samples(self, d_how_many):
        res = {}
        for label, num in d_how_many.items():
            res[label] = []
            for img in self.dataset[label][:num]:
                res[label].append(self._read_single_image(img))
        return res

    def _image(self, fname):
        img = Image.open(fname)
        img.load()
        npi = np.asarray(img, dtype="int32")
        return npi if self.resize is None \
            else skimage.transform.resize(npi, self.resize, preserve_range=True)

    def __repr__(self):
        dup = len(self.duplicates)
        total = 0
        rs = "no" if self.resize is None else self.resize
        s = f"TIGDataset (resize: {rs})\n"
        for k in self.dataset.keys():
            samples = len(self.dataset[k])
            total += samples
            s += f"  + label {k}: {samples}\n"
        s +=f"  - {dup} duplicates found, total {total} images"
        return s

if __name__ == '__main__':
    pass
