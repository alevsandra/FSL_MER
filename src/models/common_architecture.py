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

    def get_negative_labels(self, unlabel_out, position, _position, thres=0.2):
        unlabel_out = self.backbone(unlabel_out)
        r, un_idx = [], []
        softmax = nn.Softmax()

        for idx, (pos, _pos) in enumerate(zip(position, _position)):
            out = softmax(unlabel_out[idx][pos])
            if len(pos) == 1 or out.min() > thres:
                un_idx.append(idx)
                r.append(_pos[-1] if _pos else np.argmin(out.cpu().numpy(), axis=0))
            else:
                a = pos[self.get_preds(out)]
                _position[idx].append(a)
                position[idx].remove(a)
                r.append(a)

        return np.asarray(r), un_idx, unlabel_out

    def get_positive_labels(self, unlabel_out, thres=0.7):
        pseudo_labels, confident_idx = [], []
        softmax = nn.Softmax(dim=1)

        for idx, logits in enumerate(unlabel_out):
            out = softmax(logits)
            max_conf, pred = torch.max(out, dim=-1)
            if max_conf > thres:
                pseudo_labels.append(pred.item())
                confident_idx.append(idx)

        return torch.tensor(pseudo_labels), confident_idx

    def get_preds(self, out):
        return out.argmin(nn.Softmax(out), axis=0)


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

        self.loss = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy(task="multiclass", num_classes=self.num_classes)
        })
        self.threshold = threshold

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return self.optimizer

    def step(self, batch, batch_idx, tag: str):
        support, query = batch

        logits = self.protonet(support, query)
        loss = self.loss(logits, query["target"])

        position = [[i for i in range(self.num_classes)] for _ in range(len(support["audio"]))]
        _position = [[] for _ in range(len(support["audio"]))]

        pseudo_label, un_idx, logits = self.protonet.get_negative_labels(support["audio"], position, _position)
        if len(un_idx) > 0:
            loss += NegCELoss(logits[un_idx], pseudo_label) + mini_entropy_loss(logits[un_idx])

        pos_labels, pos_idx = self.protonet.get_positive_labels(support["audio"], thres=self.delta)
        if len(pos_idx) > 0:
            logits = logits[pos_idx]  # Use already computed logits for positive samples
            loss += self.loss(logits, pos_labels.to(self.device)) + mini_entropy_loss(logits[un_idx])

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
