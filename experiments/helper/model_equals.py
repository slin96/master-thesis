import numpy as np
import torch

# from experiments.helper.custom_alex import alexnet
from torchvision.models import alexnet


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


if __name__ == '__main__':
    #####
    s = 42
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # random.seed(s)
    #####

    # WARNING, FIXME: use custom alexnet
    mod1 = alexnet(pretrained=True)
    mod2 = alexnet(pretrained=True)


    result = blackbox_equals(mod1, mod1, imagenet_input)

    print(result)
