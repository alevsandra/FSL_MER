import argparse
from collections import deque
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.autograd import Variable

from model import Siamese
from dataset import PMEmo
from src.models.constants import TRAIN_CLASSES_PMEMO, TEST_CLASSES_PMEMO


# Define the Siamese LightningModule
class SiameseLightningModule(pl.LightningModule):
    def __init__(self, model, loss_fn, train_loader, test_loader, lr, save_path, test_every):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.save_path = save_path
        self.test_every = test_every
        self.test_results = deque(maxlen=20)

    def forward(self, sample1, sample2):
        return self.model(sample1, sample2)

    def training_step(self, batch, batch_idx):
        sample1, sample2, label = batch
        sample1, sample2, label = Variable(sample1), Variable(sample2), Variable(label)
        output = self.forward(sample1, sample2)
        loss = self.loss_fn(output, label)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_model(self):
        self.model.eval()
        right, error = 0, 0
        for test1, test2 in self.test_loader:
            test1, test2 = Variable(test1), Variable(test2)
            output = self.forward(test1, test2).data.cpu().numpy()
            pred = np.argmax(output)
            if pred == 0:
                right += 1
            else:
                error += 1
        accuracy = right / (right + error)
        self.test_results.append(accuracy)
        print(f"Test Accuracy: {accuracy:.4f}")
        self.model.train()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if (self.global_step + 1) % self.test_every == 0:
            self.test_model()
            avg_acc = sum(self.test_results) / len(self.test_results)
            print(f"Average Accuracy (Last {len(self.test_results)} tests): {avg_acc:.4f}")

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=self.lr)


# Main script
def main():
    parser = argparse.ArgumentParser(description="Siamese Network Training")
    parser.add_argument("--cuda", type=bool, default=True, help="use cuda")
    parser.add_argument("--train_path", type=str, default="/home/data/pin/data/omniglot/images_background",
                        help="training folder")
    parser.add_argument("--test_path", type=str, default="/home/data/pin/data/omniglot/images_evaluation",
                        help="path of testing folder")
    parser.add_argument("--way", type=int, default=20, help="how much way one-shot learning")
    parser.add_argument("--workers", type=int, default=4, help="number of dataLoader workers")
    parser.add_argument("--batch_size", type=int, default=128, help="number of batch size")
    parser.add_argument("--lr", type=float, default=0.00006, help="learning rate")
    parser.add_argument("--test_every", type=int, default=100, help="test model after each test_every iter.")
    parser.add_argument("--model_path", type=str, default="/home/data/pin/model/siamese", help="path to store model")
    parser.add_argument("--max_iter", type=int, default=50000, help="number of iterations before stopping")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('medium')

    trainSet = PMEmo(False, TRAIN_CLASSES_PMEMO, False)
    testSet = PMEmo(False, TEST_CLASSES_PMEMO, False)

    trainLoader = DataLoader(trainSet, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    testLoader = DataLoader(testSet, batch_size=args.way, shuffle=False, num_workers=args.workers)

    # Define model, loss function
    loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    model = Siamese()
    if args.cuda:
        model.cuda()

    # Create LightningModule instance
    module = SiameseLightningModule(
        model=model,
        loss_fn=loss_fn,
        train_loader=trainLoader,
        test_loader=testLoader,
        lr=args.lr,
        save_path=args.model_path,
        test_every=args.test_every
    )

    # Logger and trainer
    wandb_logger = WandbLogger(project="FSL_Siamese", job_type='train', log_model=True)
    trainer = pl.Trainer(
        max_epochs=args.max_iter // len(trainLoader),
        log_every_n_steps=1,
        logger=wandb_logger
    )

    trainer.fit(module, trainLoader)

    # Save final model
    torch.save(module.model.state_dict(), f"{args.model_path}/model-final.pt")

    # Log final accuracy
    acc = sum(module.test_results) / len(module.test_results)
    print("Final Average Accuracy:", acc)


if __name__ == "__main__":
    main()
