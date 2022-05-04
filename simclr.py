import torch
import torch.nn.functional as F

def info_nce_loss(features, temp):
    """
    Shamelessly adapted from https://github.com/sthalles/SimCLR/blob/master/simclr.py
    I believe this is the official simclr repo

    Args:
        features: torch float tensor (N,B,D)
            N is the aligned number of positive samples
            B is batch of negative samples
            D is dimensionality
        temp: float
            the temperature parameter for the softmax
    """
    device = features.get_device()
    N,B,D = features
    features = features.reshape(-1,D)
    labels = torch.cat([torch.arange(B) for i in range(N)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temp
    return logits, labels
