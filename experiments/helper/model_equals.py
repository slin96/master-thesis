import torch

# from experiments.helper.custom_alex import alexnet


def imagenet_input():
    inp = torch.rand(3, 300, 400)
    input_batch = inp.unsqueeze(0)
    return input_batch


def blackbox_equals(m1, m2, produce_input):
    inp = produce_input()

    m1.eval()
    m2.eval()

    out1 = m1(inp)
    out2 = m2(inp)

    return torch.equal(out1, out2)


def whitebox_equals(m1, m2):
    state1 = m1.state_dict()
    state2 = m2.state_dict()

    return compare_state_dicts(state1, state2)


def compare_state_dicts(d1, d2):
    for item1, item2 in zip(d1.items(), d2.items()):
        weight_tensor_1 = item1[1]
        weight_tensor_2 = item2[1]
        if not torch.equal(weight_tensor_1, weight_tensor_2):
            return False

    return True


def equals(m1, m2, produce_input):
    # whitebox and balckbox check should be redundant,
    # but this way we have an extra safety net in case we forgot a special case
    return whitebox_equals(m1, m2) and blackbox_equals(m1, m2, produce_input)
