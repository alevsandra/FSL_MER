import unittest
from src.models.MUSIC import NegCELoss, mini_entropy_loss
import mindspore.ops as ops
from mindspore import Tensor
import torch


class TestMUSICMethod(unittest.TestCase):
    def setUp(self):
        self.neg_logits = [[1.0, 2.0, 0.5, -1.0],
                           [1.5, -0.5, 0.3, 0.8],
                           [0.2, -1.0, 1.5, 2.0],
                           [-0.3, 0.7, 0.8, -1.2]]
        self.neg_labels = [1, 3, 2, 0]
        self.weight = [1.0, 1.0, 1.0, 1.0]
        self.p = [[0.25, 0.25, 0.25, 0.25],
                  [0.4, 0.3, 0.2, 0.1],
                  [0.1, 0.3, 0.4, 0.2],
                  [0.05, 0.15, 0.5, 0.3]]

    def NL_loss(self, f, labels):
        Q_1 = 1 - ops.Softmax(1)(f)
        Q = ops.Softmax(1)(Q_1)
        weight = 1 - Q
        out = weight * Q.log()
        return ops.NLLLoss()(logits=out, labels=labels, weight=Tensor(self.weight))

    def entropy_loss(self, p):
        p = ops.Softmax(axis=1)(p)
        epsilon = 1e-5
        return -1 * ops.ReduceSum()(p * ops.Log()(p + epsilon)) / p.shape[0]

    def test_NegCELoss(self):
        self.assertEqual(self.NL_loss(Tensor(self.neg_logits), Tensor(self.neg_labels))[0].item(),
                         NegCELoss(torch.Tensor(self.neg_logits),
                                   torch.Tensor(self.neg_labels).type(torch.LongTensor)).item(),
                         "The NegCELoss is wrong")

    def test_entropy_loss(self):
        self.assertEqual(self.entropy_loss(Tensor(self.p)),
                         mini_entropy_loss(torch.Tensor(self.p)).item(),
                         "The entropy loss is wrong")
