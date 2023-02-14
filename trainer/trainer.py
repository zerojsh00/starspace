import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

class StarSpaceTrainer:

    def __init__(self, model, X_train, y_train, X_test, y_test, optimizer, n_epochs, save_dir, device, featurizer, patience):
        self.model = model
        self.X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        self.y_train = torch.tensor(y_train, dtype=torch.long, device=device)
        self.X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
        self.y_test = torch.tensor(y_test, dtype=torch.long, device=device)
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.save_dir = save_dir
        self.featurizer = featurizer
        self.train_loss = []
        self.train_f1 = []
        self.test_loss = []
        self.test_f1 = []
        self.early_stopping = EarlyStopping(patience=patience, verbose=False)


    def _train_epoch(self):
        self.model.train()

        # future works : need to implement dataloader for batch training
        output = self.model(self.X_train, self.y_train)
        loss = output['loss']
        f1 = f1_score(self.y_train.detach().cpu(), output['prediction'].detach().cpu(), average='macro')
        loss.backward()
        self.optimizer.step()

        self.train_loss.append(loss.detach().cpu())
        self.train_f1.append(f1)

    def _validate(self):
        self.model.eval()

        with torch.no_grad():
            output = self.model(self.X_test, self.y_test)
            loss = output['loss']
            f1 = f1_score(self.y_test.detach().cpu(), output['prediction'].detach().cpu(), average='macro')
            self.test_loss.append(loss.detach().cpu())
            self.test_f1.append(f1)
            self.early_stopping(loss.detach().cpu(), self.model)


        return output['prediction']

    def _plot_training(self):
        plt.plot(self.train_loss, label="Training Loss")
        plt.plot(self.test_loss, label="Test Loss")
        plt.plot(self.train_f1, label="Training f1")
        plt.plot(self.test_f1, label="Test f1")
        plt.legend()
        plt.grid()
        plt.savefig('./training_summary_{}.png'.format(self.featurizer), dpi=300)

    def fit(self):
        for epoch in tqdm(range(self.n_epochs)):
            self._train_epoch()
            y_pred = self._validate()
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
        torch.save(self.model.state_dict(), self.save_dir+"/SR_model_{}.pt".format(self.featurizer))
        self._plot_training()
        return y_pred


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss