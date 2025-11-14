# utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLossWithMining(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLossWithMining, self).__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        pdist = torch.cdist(embeddings, embeddings, p=2)

        loss = 0.0
        count = 0
        for i in range(len(embeddings)):
            anchor_label = labels[i]
            dists = pdist[i]

            pos_mask = (labels == anchor_label) & (torch.arange(len(labels), device=labels.device) != i)
            neg_mask = (labels != anchor_label)

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            hardest_pos = dists[pos_mask].max()
            semi_hard_negs = dists[neg_mask]
            mask = semi_hard_negs > hardest_pos

            if mask.sum() == 0:
                continue  # Skip if no valid semi-hard negatives

            semi_hard_neg = semi_hard_negs[mask].min()
            triplet_loss = F.relu(hardest_pos - semi_hard_neg + self.margin)

            loss += triplet_loss
            count += 1

        return loss / count if count > 0 else torch.tensor(0.0, requires_grad=True).to(embeddings.device)
