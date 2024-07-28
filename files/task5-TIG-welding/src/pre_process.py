import misc

def main():
    topdir_dataset = "~/1/dataset/al5083"
    topdir_preprocess = "~/1/dataset/preprocess"
    json_files = [
        ("train/train.json", "train"),
        ("test/test.json", "test")
]
    rsize = (60,50)
    normalize = True
    da = misc.TIGDataAccess(dir_in=topdir_dataset,
                            how_many_classes=6,
                            dir_preprocess=topdir_preprocess,
                            json_files=json_files)
    da.setup(max_items_per_label=0, resize=rsize, normalize=normalize)

if __name__ == '__main__':
    main()
