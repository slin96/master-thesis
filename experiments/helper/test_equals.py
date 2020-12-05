import unittest

import torch
from torchvision import models

from experiments.helper.deterministic import set_deterministic
from experiments.helper.model_equals import blackbox_equals, imagenet_input


class TestModelEquals(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        torch._set_deterministic(True)

    def test_blackbox_resnet18_pretrained(self):
        mod1 = models.resnet18(pretrained=True)
        mod2 = models.resnet18(pretrained=True)

        self.assertTrue(blackbox_equals(mod1, mod2, imagenet_input))

    def test_blackbox_resnet50_pretrained(self):
        mod1 = models.resnet50(pretrained=True)
        mod2 = models.resnet50(pretrained=True)

        self.assertTrue(blackbox_equals(mod1, mod2, imagenet_input))

    def test_blackbox_resnet152_pretrained(self):
        mod1 = models.resnet152(pretrained=True)
        mod2 = models.resnet152(pretrained=True)

        self.assertTrue(blackbox_equals(mod1, mod2, imagenet_input))

    def test_blackbox_vgg19_pretrained(self):
        mod1 = models.vgg19(pretrained=True)
        mod2 = models.vgg19(pretrained=True)

        self.assertTrue(blackbox_equals(mod1, mod2, imagenet_input))

    def test_blackbox_alexnet_pretrained(self):
        mod1 = models.alexnet(pretrained=True)
        mod2 = models.alexnet(pretrained=True)

        self.assertTrue(blackbox_equals(mod1, mod2, imagenet_input))

    def test_blackbox_resnet18_resnet152_pretrained(self):
        mod1 = models.resnet18(pretrained=True)
        mod2 = models.resnet152(pretrained=True)

        self.assertFalse(blackbox_equals(mod1, mod2, imagenet_input))

    def test_blackbox_not_pretrained(self):
        mod1 = models.resnet18()
        mod2 = models.resnet18()

        # we expect this to be false since the weight initialization is random
        self.assertFalse(blackbox_equals(mod1, mod2, imagenet_input))

    def test_blackbox_resnet18_not_pretrained_deterministic(self):
        set_deterministic()
        mod1 = models.resnet18()

        set_deterministic()
        mod2 = models.resnet18()

        # we expect this to be true, the weights are randomly initialized,
        # but we set the seeds before weight initialization
        self.assertTrue(blackbox_equals(mod1, mod2, imagenet_input))

    def test_blackbox_resnet152_not_pretrained_deterministic(self):
        set_deterministic()
        mod1 = models.resnet152()

        set_deterministic()
        mod2 = models.resnet152()

        # we expect this to be true, the weights are randomly initialized,
        # but we set the seeds before weight initialization
        self.assertTrue(blackbox_equals(mod1, mod2, imagenet_input))

    def test_blackbox_vgg19_not_pretrained_deterministic(self):
        set_deterministic()
        mod1 = models.vgg19()

        set_deterministic()
        mod2 = models.vgg19()

        # we expect this to be true, the weights are randomly initialized,
        # but we set the seeds before weight initialization
        self.assertTrue(blackbox_equals(mod1, mod2, imagenet_input))

    def test_blackbox_alexnet_not_pretrained_deterministic(self):
        set_deterministic()
        mod1 = models.alexnet()

        set_deterministic()
        mod2 = models.alexnet()

        # we expect this to be true, the weights are randomly initialized,
        # but we set the seeds before weight initialization
        self.assertTrue(blackbox_equals(mod1, mod2, imagenet_input))
