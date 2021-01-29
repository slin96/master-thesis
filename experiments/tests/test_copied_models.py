import unittest

from mmlib.deterministic import set_deterministic
from mmlib.helper import imagenet_input
from mmlib.equal import model_equal
from torchvision import models

from experiments.models.googlenet import googlenet
from experiments.models.mobilenet import mobilenet_v2
from experiments.models.resnet152 import resnet152
from experiments.models.resnet18 import resnet18
from experiments.models.resnet50 import resnet50


class TestCopiedModels(unittest.TestCase):

    def test_resnet18(self):
        self._test_initialized_equals(models.resnet18, resnet18)
        self._test_pretrained_equals(models.resnet18, resnet18)

    def test_resnet50(self):
        self._test_initialized_equals(models.resnet50, resnet50)
        self._test_pretrained_equals(models.resnet50, resnet50)

    def test_resnet152(self):
        self._test_initialized_equals(models.resnet152, resnet152)
        self._test_pretrained_equals(models.resnet152, resnet152)

    def test_googlenet(self):
        self._test_initialized_equals(models.googlenet, googlenet)
        self._test_pretrained_equals(models.googlenet, googlenet)

    def test_mobilenet_v2(self):
        self._test_initialized_equals(models.mobilenet_v2, mobilenet_v2)
        self._test_pretrained_equals(models.mobilenet_v2, mobilenet_v2)

    def _test_pretrained_equals(self, m1, m2):
        local_net = m1(pretrained=True)
        lib_net = m2(pretrained=True)

        self.assertTrue(model_equal(local_net, lib_net, imagenet_input))

    def _test_initialized_equals(self, m1, m2):
        set_deterministic()
        local_net = m1()

        set_deterministic()
        lib_net = m2()

        self.assertTrue(model_equal(local_net, lib_net, imagenet_input))
