import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class StarSpace(nn.Module):
    def __init__(self, X_dim, n_labels, emb_dim=32, drop_rate=0.5):
        super().__init__()

        # feature들에 대한 임베딩
        self.feature_emb_layer = nn.Sequential(
            nn.Linear(X_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Linear(64, emb_dim)
        ).to(device)

        # 레이블에 대한 임베딩
        self.label_emb = nn.Embedding(n_labels, emb_dim, max_norm=10.0).to(device)
        self.label_emb_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        ).to(device)

        # loss
        self.ce_loss = nn.CrossEntropyLoss()  # Softmax + CrossEntropyLoss
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X, y):
        # 임베딩 레이어를 통과하여 임베딩을 획득
        feature_emb = self.feature_emb_layer(X)

        # 어차피 positive target 임베딩과 negative target 임베딩이 함께 있음
        y_embs = self.label_emb_layer(self.label_emb.weight)

        # 정답에 대한 레이블의 스코어와 negative 레이블의 스코어가 함께 계산됨
        sim_scores = torch.matmul(feature_emb, y_embs.transpose(0, 1))

        loss = self.ce_loss(sim_scores, y)
        confidence, prediction = torch.max(self.softmax(sim_scores), dim=-1)

        return {"loss": loss, "prediction": prediction, "confidence": confidence}