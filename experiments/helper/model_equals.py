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
