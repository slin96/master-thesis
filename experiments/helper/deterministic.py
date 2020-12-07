import torch

SEED = 42


def deterministic(func, f_args, f_kwargs):
    # TODO check if data loaders are determinitsic
    # TODO maybe print warning for multiGPU
    set_deterministic()
    return func(*f_args, **f_kwargs)


def set_deterministic():
    # TODO maybe in the future we also have to set seed for used libraries, e.g. numpy
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # TODO check if we can solve this more nicely
    # TODO check what to do for deterministic on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
