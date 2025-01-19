import pickle
import time
from collections import deque

import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torchmetrics import Accuracy

from model import Siamese
from dataset import PMEmo
from src.models.constants import TRAIN_CLASSES_PMEMO, TEST_CLASSES_PMEMO


class SiameseNetworkLearner(pl.LightningModule):
    def __init__(self,
                 net: nn.Module,
                 num_classes,
                 learning_rate: float = 1e-3,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.net = net
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy(task="multiclass", num_classes=self.num_classes)
        })

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return self.optimizer

    def step(self, batch, batch_idx, tag: str):
        support, query = batch

        logits = self.net(support, query)
        loss = self.loss(logits, query["target"])

        output = {"loss": loss}
        for k, metric in self.metrics.items():
            output[k] = metric(logits, query["target"])

        for k, v in output.items():
            self.log(f"{k}/{tag}", v)
        return output

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Siamese Network Training")
    parser.add_argument("--cuda", type=bool, default=True, help="use cuda")
    parser.add_argument("--train_path", type=str, default="/home/data/pin/data/omniglot/images_background",
                        help="training folder")
    parser.add_argument("--test_path", type=str, default="/home/data/pin/data/omniglot/images_evaluation",
                        help="path of testing folder")
    parser.add_argument("--way", type=int, default=20, help="how much way one-shot learning")
    parser.add_argument("--times", type=int, default=400, help="number of samples to test accuracy")
    parser.add_argument("--workers", type=int, default=4, help="number of dataLoader workers")
    parser.add_argument("--batch_size", type=int, default=128, help="number of batch size")
    parser.add_argument("--lr", type=float, default=0.00006, help="learning rate")
    parser.add_argument("--show_every", type=int, default=10, help="show result after each show_every iter.")
    parser.add_argument("--save_every", type=int, default=100, help="save model after each save_every iter.")
    parser.add_argument("--test_every", type=int, default=100, help="test model after each test_every iter.")
    parser.add_argument("--max_iter", type=int, default=50000, help="number of iterations before stopping")
    parser.add_argument("--model_path", type=str, default="/home/data/pin/model/siamese", help="path to store model")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="gpu ids used to train")

    args = parser.parse_args()

    data_transforms = transforms.Compose([
        transforms.RandomAffine(15),
        transforms.ToTensor()
    ])

    trainSet = PMEmo(False, TRAIN_CLASSES_PMEMO, False)
    testSet = PMEmo(False, TEST_CLASSES_PMEMO, False)

    testLoader = DataLoader(testSet, batch_size=args.way, shuffle=False, num_workers=args.workers)
    trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean')
    net = Siamese()
    net.cuda()

    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    optimizer.zero_grad()

    train_loss = []
    loss_val = 0
    time_start = time.time()
    queue = deque(maxlen=20)

    for batch_id, (sample1, sample2, label) in enumerate(trainLoader, 1):
        sample1, sample2, label = Variable(sample1.cuda()), Variable(sample2.cuda()), Variable(label.cuda())
        optimizer.zero_grad()
        output = net.forward(sample1, sample2)
        loss = loss_fn(output, label)
        loss_val += loss.item()
        loss.backward()
        optimizer.step()
        if batch_id % args.show_every == 0:
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
            batch_id, loss_val / args.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        if batch_id % args.save_every == 0:
            torch.save(net.state_dict(), args.model_path + '/model-inter-' + str(batch_id + 1) + ".pt")
        if batch_id % args.test_every == 0:
            right, error = 0, 0
            for _, (test1, test2) in enumerate(testLoader, 1):
                if args.cuda:
                    test1, test2 = test1.cuda(), test2.cuda()
                test1, test2 = Variable(test1), Variable(test2)
                output = net.forward(test1, test2).data.cpu().numpy()
                pred = np.argmax(output)
                if pred == 0:
                    right += 1
                else:
                    error += 1
            print('*' * 70)
            print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f' % (
            batch_id, right, error, right * 1.0 / (right + error)))
            print('*' * 70)
            queue.append(right * 1.0 / (right + error))
        train_loss.append(loss_val)
    #  learning_rate = learning_rate * 0.95

    with open('train_loss.txt', 'wb') as f:
        pickle.dump(train_loss, f)

    acc = 0.0
    for d in queue:
        acc += d
    print("#" * 70)
    print("final accuracy: ", acc / 20)
