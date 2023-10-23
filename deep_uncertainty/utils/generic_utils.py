import yaml

def get_yaml(fpath):
    with open(fpath, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg