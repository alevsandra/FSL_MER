from torch import nn
import torch
import pytorch_lightning as pl
import numpy as np
from torchmetrics import Accuracy
from MUSIC import NegCELoss, mini_entropy_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class PrototypicalNet(nn.Module):

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, support: dict, query: dict):
        """
        Forward pass through the protonet.

        Args:
            support (dict): A dictionary containing the support set.
                The support set dict must contain the following keys:
                    - audio: A tensor of shape (n_support, n_channels, n_samples)
                    - label: A tensor of shape (n_support) with label indices
                    - classlist: A tensor of shape (n_classes) containing the list of classes in this episode
            query (dict): A dictionary containing the query set.
                The query set dict must contain the following keys:
                    - audio: A tensor of shape (n_query, n_channels, n_samples)

        Returns:
            logits (torch.Tensor): A tensor of shape (n_query, n_classes) containing the logits

        After the forward pass, the support dict is updated with the following keys:
            - embeddings: A tensor of shape (n_support, n_features) containing the embeddings
            - prototypes: A tensor of shape (n_classes, n_features) containing the prototypes

        The query dict is updated with
            - embeddings: A tensor of shape (n_query, n_features) containing the embeddings

        """
        # compute the embeddings for the support and query sets
        support["embeddings"] = self.backbone(support["audio"])
        query["embeddings"] = self.backbone(query["audio"])

        # group the support embeddings by class
        support_embeddings = []
        for idx in range(len(support["classlist"])):
            embeddings = support["embeddings"][support["target"] == idx]
            support_embeddings.append(embeddings)
        support_embeddings = torch.stack(support_embeddings)

        # compute the prototypes for each class
        prototypes = support_embeddings.mean(dim=1)
        support["prototypes"] = prototypes

        # print("Prototypes Shape: ", prototypes.shape)
        # print("Embeddings Shape: ", query["embeddings"].shape)
        # compute the distances between each query and prototype
        distances = torch.cdist(
            query["embeddings"].unsqueeze(0),
            prototypes.unsqueeze(0),
            p=2
        ).squeeze(0)

        # square the distances to get the sq euclidean distance
        distances = distances ** 2
        logits = -distances

        # return the logits
        return logits


class FewShotLearner(pl.LightningModule):
    def __init__(self,
                 protonet: nn.Module,
                 num_classes,
                 learning_rate: float = 1e-3,
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.protonet = protonet
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.loss = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy(task="multiclass", num_classes=self.num_classes)
        })

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return self.optimizer

    def step(self, batch, batch_idx, tag: str):
        support, query = batch

        logits = self.protonet(support, query)
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


class FewShotNegativeLearner(pl.LightningModule):
    def __init__(self,
                 protonet: nn.Module,
                 num_classes,
                 learning_rate: float = 1e-3,
                 threshold: float = 0.2):
        super().__init__()
        self.save_hyperparameters()
        self.protonet = protonet
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.optimizer_NL = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        self.loss = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy(task="multiclass", num_classes=self.num_classes)
        })
        self.threshold = threshold

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return self.optimizer

    def train_loop(self, inputs, targets):
        def forward_fn(logits, label):
            label = label.detach().to(device='cuda')
            return NegCELoss(logits, label) + mini_entropy_loss(logits), logits

        def train_step(data, label):
            loss, logits = forward_fn(data, label)
            self.optimizer_NL.zero_grad()
            if not loss.requires_grad:
                loss.requires_grad = True
            loss.backward()
            self.optimizer_NL.step()
            return loss, logits

        return train_step(torch.tensor(inputs), torch.tensor(targets))

    def step(self, batch, batch_idx, tag: str):
        support, query = batch

        logits = self.protonet(support, query)
        loss = self.loss(logits, query["target"])

        position = [[i for i in range(self.num_classes)] for _ in range(len(query["embeddings"]))]
        _position = [[] for _ in range(len(query["embeddings"]))]

        pseudo_label, unselected_indices, neg_logits = self.get_negative_labels(query["embeddings"], position, _position)
        selected_indices = [idx for idx in range(len(query["embeddings"])) if idx not in unselected_indices]

        if selected_indices:
            for epoch in range(10):
                neg_loss, logits_xd = self.train_loop(query["embeddings"][selected_indices], pseudo_label)
                print(f"Epoch: {epoch}  Loss: {neg_loss}")

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

    def get_negative_labels(self, unlabel_out, position, _position, thres=0.2):
        results, uncertain_indices = [], []
        softmax = nn.Softmax()

        for idx, (pos, _pos) in enumerate(zip(position, _position)):
            out = softmax(unlabel_out[idx][pos])
            if len(pos) == 1:
                uncertain_indices.append(idx)
                continue
            if out.min() > thres:
                uncertain_indices.append(idx)
                results.append(_pos[-1] if _pos else torch.argmin(out).item())
                continue

            a = pos[self.get_preds(out)]
            _position[idx].append(a)
            position[idx].remove(a)
            results.append(a)

        return np.asarray(results), uncertain_indices, unlabel_out

    def get_preds(self, out):
        return np.argmin(nn.Softmax(out), axis=0)
