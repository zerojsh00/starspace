import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
class StarSpaceTrainer:
    def __init__(self, model, X_train, y_train, X_test, y_test, optimizer, n_epochs, save_dir, device):
        self.model = model
        self.X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        self.y_train = torch.tensor(y_train, dtype=torch.long, device=device)
        self.X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
        self.y_test = torch.tensor(y_test, dtype=torch.long, device=device)
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.save_dir = save_dir
        self.train_loss = []
        self.train_f1 = []
        self.test_loss = []
        self.test_f1 = []
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
        return output['prediction']
    def _plot_training(self):
        plt.plot(self.train_loss, label="Training Loss")
        plt.plot(self.test_loss, label="Test Loss")
        plt.plot(self.train_f1, label="Training f1")
        plt.plot(self.test_f1, label="Test f1")
        plt.legend()
        plt.grid()
        plt.savefig('{}/training_summary.png'.format(self.save_dir), dpi=300)
    def fit(self):
        for epoch in range(self.n_epochs):
            self._train_epoch()
            y_pred = self._validate()
        torch.save(self.model.state_dict(), self.save_dir+"/model.pt")
        self._plot_training()
        return y_pred