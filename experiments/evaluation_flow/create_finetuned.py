import torch
from mmlib.equal import state_dict_equal
from mmlib.util.helper import get_device


def _get_fine_tuned_model(model_class, base_snapshot, copy_snapshot, layer_names):
    device = get_device(None)

    base = torch.load(base_snapshot, map_location=device)
    copy = torch.load(copy_snapshot, map_location=device)
    new = torch.load(base_snapshot, map_location=device)

    for layer_name in layer_names:
        new[layer_name] = copy[layer_name]

    assert not state_dict_equal(base, new)
    assert not state_dict_equal(copy, new)
    assert not state_dict_equal(copy, base)

    model: torch.nn.Module = model_class()
    model.load_state_dict(new)

    return model


def get_fine_tuned_model(model_class, base_snapshot, copy_snapshot):
    class_name = model_class.__name__
    class_name = class_name.lower()

    if 'res' in class_name or 'google' in class_name:
        return _get_fine_tuned_model(model_class, base_snapshot, copy_snapshot, ['fc.weight', 'fc.bias'])
    else:
        return _get_fine_tuned_model(model_class, base_snapshot, copy_snapshot,
                                     ['classifier.1.weight', 'classifier.1.bias'])
