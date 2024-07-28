import numpy as np
import misc

topdir_dataset = "~/1/dataset/al5083"
json_files = [
    ("train/train.json", "train"),
    ("test/test.json", "test")
]

def test(rsize=None, details=True, normalize=True):
    td = misc.TIGDataset(topdir=topdir_dataset,
                         resize=rsize,
                         json_files=json_files)
    if details:
        print(td)
    samples = td.samples({0:10, 1:11, 5:15}, normalize=normalize)

    for k in samples:
        fst = samples[k][0]
        print(f"label {k}: {len(samples[k])} images")
        print("Shape:", fst.shape)
        print("Sum:", np.sum(fst))
        print("Max:", np.max(fst),
              ", Min:", np.min(fst),
              ", Mean:", np.sum(fst)/fst.size)

def main():
    for nm in [False, True]:
        print(f"---------------------- normalize = {nm} ---------------------")
        print("==== No compression ====")
        test(normalize=nm)
        print("==== With compression ====")
        test((60,50), details=False, normalize=nm)

if __name__ == '__main__':
    main()
