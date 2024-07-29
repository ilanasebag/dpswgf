import yaml

class DotDict(dict):
    """
    Dot notation access to nested dictionary attributes.
    """
    def __init__(self, *kargs, replace_nones: bool = False, **kwargs):
        super().__init__(*kargs, **kwargs)
        for k, v in self.items():
            if replace_nones and v is None:
                self[k] = DotDict()
            if isinstance(v, dict):
                self[k] = DotDict(v, replace_nones=replace_nones)
            if isinstance(v, list):
                self[k] = [DotDict(w, replace_nones=replace_nones) if isinstance(w, dict) else w for w in v]

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_yaml(path):
    """
    Loads a yaml input file.
    """
    with open(path, 'r') as f:
        opt = yaml.safe_load(f)
    return DotDict(opt)

