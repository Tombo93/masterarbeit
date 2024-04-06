import os


def make_data_dirs():
    """Skript for creating data dirs in this application"""
    root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir
        )
    )
    data_dir = os.path.join(root, "data")
    raw_dir = os.path.join(data_dir, "raw")
    interim_dir = os.path.join(data_dir, "interim")
    processed_dir = os.path.join(data_dir, "processed")

    if os.path.isdir(data_dir):
        print("Data dir already exists")
        return True

    print("Making data dir...")
    os.mkdir(data_dir)
    os.mkdir(raw_dir)
    os.mkdir(interim_dir)
    os.mkdir(processed_dir)
    for parent_dir in [raw_dir, interim_dir, processed_dir]:
        os.mkdir(os.path.join(parent_dir, "isic"))
        os.mkdir(os.path.join(parent_dir, "cifar10"))




if __name__ == "__main__":
    make_data_dirs()
