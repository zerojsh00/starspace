import torch
import torch.optim as optim

import numpy as np
import argparse
import random
from data.dataset import Dataset
from model.starspace import StarSpace
from trainer.trainer import StarSpaceTrainer


def _get_parser():
    parser = argparse.ArgumentParser(description='NLP Tasks - Named Entity Recognition')
    # Basic arguments
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='model')
    # Model arguments
    parser.add_argument('--tokenizer_model', type=str, default='klue/bert-base')
    parser.add_argument('--vectorizer_name', type=str, default='TfidfVectorizer')
    # Dataset arguments
    parser.add_argument('--dataset_name', type=str, default='data/intent_data.csv')
    # Training arguments
    parser.add_argument('--test_size', type=float, default=0.3)
    # parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--n_epochs', type=int, default=250)

    return parser

def main():

    parser = _get_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get dataset
    ((X_train, X_test, y_train, y_test, indices_train, indices_test), n_labels, categ2label) = Dataset(args).load_dataset()

    # future works : need to implement dataloader for batch training

    # get model
    model = StarSpace(X_train.shape[1], n_labels)

    # train model
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    y_pred = StarSpaceTrainer(model, X_train, y_train, X_test, y_test, optimizer, args.n_epochs, args.save_dir, device).fit()
    print("정답 : " + ' | '.join(([categ2label[dep] for dep in y_test])))
    print("예측 : " + ' | '.join(([categ2label[dep.item()] for dep in y_pred])))


if __name__ == '__main__':
    main()