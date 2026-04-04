import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AAMSoftmax(nn.Module):
    """
    Additive Angular Margin Softmax for speaker classification
    """
    def __init__(self, embedding_dim, num_classes, margin=0.2, scale=30):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch, embedding_dim)
            labels: (batch)
        """
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, dim=1)
        self.weight.data = F.normalize(self.weight.data, dim=1)
        
        # Cosine similarity
        cos_theta = F.linear(embeddings, self.weight)
        
        # Apply margin
        theta = torch.acos(torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7))
        margin_theta = theta + self.margin
        cos_margin_theta = torch.cos(margin_theta)
        
        # Create mask for target classes
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Apply margin only to target classes
        output = (one_hot * cos_margin_theta) + ((1 - one_hot) * cos_theta)
        output *= self.scale
        
        return F.cross_entropy(output, labels)

class ContrastiveLoss(nn.Module):
    """
    SimCLR-style contrastive loss for speaker embeddings
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch, embedding_dim)
            labels: (batch)
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Cosine similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create mask for positive pairs (same speaker)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove self-similarity
        mask.fill_diagonal_(0)
        
        # Compute loss
        exp_sim = torch.exp(similarity)
        
        # Positive pairs
        pos_sum = torch.sum(mask * exp_sim, dim=1)
        
        # All pairs (including negatives)
        all_sum = torch.sum(exp_sim, dim=1) - torch.diag(exp_sim)
        
        # Loss for each sample
        loss = -torch.log(pos_sum / (all_sum + 1e-8))
        
        # Only consider samples with positive pairs
        valid = (pos_sum > 0).float()
        loss = torch.sum(loss * valid) / (valid.sum() + 1e-8)
        
        return loss

class GE2ELoss(nn.Module):
    """
    Generalized End-to-End Loss for speaker verification
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch, embedding_dim)
            labels: (batch)
        """
        # Get unique speakers
        unique_speakers = torch.unique(labels)
        num_speakers = len(unique_speakers)
        num_utterances = len(labels) // num_speakers
        
        # Reshape to (speakers, utterances, embedding_dim)
        embeddings = embeddings.view(num_speakers, num_utterances, -1)
        
        # Centroids (excluding each utterance)
        centroids = []
        for i in range(num_speakers):
            speaker_embeds = embeddings[i]
            centroid = speaker_embeds.mean(dim=0)
            centroids.append(centroid)
        centroids = torch.stack(centroids)
        
        # Loss computation
        loss = 0
        for i in range(num_speakers):
            for j in range(num_utterances):
                # Positive similarity
                pos_sim = F.cosine_similarity(embeddings[i, j], centroids[i], dim=0)
                
                # Negative similarities
                neg_sims = []
                for k in range(num_speakers):
                    if k != i:
                        neg_sim = F.cosine_similarity(embeddings[i, j], centroids[k], dim=0)
                        neg_sims.append(neg_sim)
                
                # Loss for this utterance
                loss += -pos_sim + torch.logsumexp(torch.stack(neg_sims), dim=0)
        
        loss = loss / (num_speakers * num_utterances)
        return loss

class MultiLoss(nn.Module):
    """
    Combined loss: AAM-Softmax + Contrastive + GE2E
    """
    def __init__(self, config, num_speakers):
        super().__init__()
        self.aam = AAMSoftmax(config.FINAL_EMBED_DIM, num_speakers, 
                              margin=config.AAM_MARGIN, scale=config.AAM_SCALE)
        self.contrastive = ContrastiveLoss(temperature=config.CONTRASTIVE_TEMPERATURE)
        self.ge2e = GE2ELoss()
        
        self.contrastive_weight = config.CONTRASTIVE_WEIGHT
        self.ge2e_weight = config.GE2E_WEIGHT
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (batch, embedding_dim)
            labels: (batch)
        """
        # AAM-Softmax loss
        aam_loss = self.aam(embeddings, labels)
        
        # Contrastive loss
        contrastive_loss = self.contrastive(embeddings, labels)
        
        # GE2E loss
        ge2e_loss = self.ge2e(embeddings, labels)
        
        # Combined loss
        total_loss = aam_loss + self.contrastive_weight * contrastive_loss + self.ge2e_weight * ge2e_loss
        
        return {
            'total': total_loss,
            'aam': aam_loss,
            'contrastive': contrastive_loss,
            'ge2e': ge2e_loss
        }