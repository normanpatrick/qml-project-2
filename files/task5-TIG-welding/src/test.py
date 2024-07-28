import numpy as np
import misc

topdir_dataset = "~/1/dataset/al5083"
topdir_preprocess = "~/1/dataset/preprocess"
json_files = [
    ("train/train.json", "train"),
    ("test/test.json", "test")
]

def _helper_01(rsize=None, details=True, normalize=True):
    td = misc.TIGDataset(topdir=topdir_dataset,
                         resize=rsize,
                         normalize=normalize,
                         how_many_classes=6,
                         json_files=json_files)
    if details:
        print(td)
    samples = td.samples({0:10, 1:11, 5:15})

    for k in samples:
        fst = samples[k][0]
        print(f"label {k}: {len(samples[k])} images")
        print("Shape:", fst.shape)
        print("Sum:", np.sum(fst))
        print("Max:", np.max(fst),
              ", Min:", np.min(fst),
              ", Mean:", np.sum(fst)/fst.size)

def test_01():
    for nm in [False, True]:
        print(f"---------------------- normalize = {nm} ---------------------")
        print("==== No compression ====")
        _helper_01(normalize=nm)
        print("==== With compression ====")
        _helper_01((60,50), details=False, normalize=nm)

def test_02():
    rsize = (60,50)
    normalize = True
    da = misc.TIGDataAccess(dir_in=topdir_dataset,
                            how_many_classes=6,
                            dir_preprocess=topdir_preprocess,
                            json_files=json_files)
    da.setup(max_items_per_label=5, resize=rsize, normalize=normalize)

def main():
    test_01()
    test_02()

if __name__ == '__main__':
    main()
