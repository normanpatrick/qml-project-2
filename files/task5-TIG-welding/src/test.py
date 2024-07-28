import misc

topdir_dataset = "~/1/dataset/al5083"
json_files = [
    ("train/train.json", "train"),
    ("test/test.json", "test")
]

def test(rsize=None, details=True):
    td = misc.TIGDataset(topdir=topdir_dataset,
                         resize=rsize,
                         json_files=json_files)
    if details:
        print(td)
    samples = td.samples({0:10, 1:11, 5:15})

    for k in samples:
        print(f"label {k}: {len(samples[k])} images")
        print(samples[k][0].shape)

def main():
    print("==== No compression ====")
    test()
    print("==== With compression ====")
    test((60,50), details=False)

if __name__ == '__main__':
    main()
