from torch import nn
import torch
import pytorch_lightning as pl
import numpy as np
from torchmetrics import Accuracy
from MUSIC import NegCELoss, mini_entropy_loss
from tqdm import tqdm


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
        data, label = torch.tensor(inputs), torch.tensor(targets)
        label = label.detach().to(device='cuda')
        loss = NegCELoss(data, label) + mini_entropy_loss(data)
        self.optimizer_NL.zero_grad()
        if not loss.requires_grad:
            loss.requires_grad = True
        loss.backward()
        self.optimizer_NL.step()
        return loss, data

    def step(self, batch, batch_idx, tag: str):
        support, query = batch
        logits = self.protonet(support, query)
        loss = self.loss(logits, query["target"])

        if tag is None:
            # Extract query embeddings and prototypes
            query_embeddings = query["embeddings"]
            prototypes = support["prototypes"]

            # Pseudo-label the query embeddings
            self.refine_with_pseudo_labels(query_embeddings, prototypes)

            loss = NegCELoss(logits, query["target"]) + mini_entropy_loss(logits)

        # position = [[i for i in range(self.num_classes)] for _ in range(len(query["embeddings"]))]
        # _position = [[] for _ in range(len(query["embeddings"]))]
        #
        # pseudo_label, unselected_indices, neg_logits = self.get_negative_labels(query["embeddings"], position, _position)
        # selected_indices = [idx for idx in range(len(query["embeddings"])) if idx not in unselected_indices]
        #
        # if selected_indices:
        #     for epoch in (pbar := tqdm(range(10))):
        #         neg_loss, logits_xd = self.train_loop(query["embeddings"][selected_indices], pseudo_label)
        #         pbar.set_description(f"Epoch: {epoch}  Loss: {neg_loss}")

        output = {"loss": loss}
        for k, metric in self.metrics.items():
            output[k] = metric(logits, query["target"])

        for k, v in output.items():
            self.log(f"{k}/{tag}", v)
        return output

    def training_step(self, batch, batch_idx, tag: str):
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

    def generate_negative_pseudo_labels(self, query_embeddings, prototypes, threshold=0.2):
        """
        Generate negative pseudo-labels for the query set based on the MUSIC approach.
        The negative pseudo-label is given to the class with the minimum predicted probability.
        """
        distances = torch.cdist(query_embeddings.unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze(0)
        distances = distances ** 2  # squared Euclidean distance

        # Compute the negative probabilities for each query embedding
        neg_probabilities = torch.min(distances, dim=1).values
        neg_class = torch.argmin(distances, dim=1)

        # Pseudo-labeling based on the negative class
        pseudo_labels = torch.zeros_like(neg_probabilities)
        confidence_mask = neg_probabilities <= threshold
        pseudo_labels[confidence_mask] = neg_class[confidence_mask]

        return pseudo_labels, confidence_mask

    def pseudo_labeling_step(self, query_embeddings, prototypes):
        """
        Generate pseudo-labels and update the training data.
        For each sample, we exclude the class with the most confident negative prediction.
        """
        pseudo_labels, confidence_mask = self.generate_negative_pseudo_labels(query_embeddings, prototypes)

        pseudo_labeled_data = []
        pseudo_labeled_targets = []

        for i, label in enumerate(pseudo_labels):
            if confidence_mask[i]:  # Only include those with confident pseudo-labels
                pseudo_labeled_data.append(query_embeddings[i])
                pseudo_labeled_targets.append(label)

        return torch.stack(pseudo_labeled_data), torch.tensor(pseudo_labeled_targets)

    def refine_with_pseudo_labels(self, query_embeddings, prototypes):
        """
        Refine the model by including pseudo-labeled data in the training process.
        """
        pseudo_labeled_data, pseudo_labeled_targets = self.pseudo_labeling_step(query_embeddings, prototypes)

        if len(pseudo_labeled_data) > 0:
            support_data = torch.cat([query_embeddings, pseudo_labeled_data])
            support_targets = torch.cat([query_embeddings["target"], pseudo_labeled_targets])
            support = {"audio": support_data, "target": support_targets}

            # Retrain the model using the augmented dataset
            self.training_step(support, 0, tag="retrain")
