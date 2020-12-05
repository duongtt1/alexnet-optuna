from typing import Any, Callable, List, Tuple
import optuna  # type: ignore
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore
from torchvision.datasets import ImageFolder , MNIST # type: ignore
import json
from optuna.visualization import plot_optimization_history


device = "cuda" if torch.cuda.is_available() else "cpu"

def create_loader(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    train_set = ImageFolder(root='/content/fire_dataset/train', transform=data_transforms['train'])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = ImageFolder(root='/content/fire_dataset/val',transform=data_transforms['val'])
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

class AlexNet(nn.Module):
    def __init__(self, trial: optuna.trial.Trial, num_classes: int = 2) -> None:
        super(AlexNet, self).__init__()
        #features
        self.activation = getattr(F, trial.suggest_categorical('activation', ['relu', 'elu']))
        self.maxpool_k3_s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        #avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        #classifier
        self.dropout1 = nn.Dropout2d(p=trial.suggest_uniform("dropout_param_1", 0.1, 0.9))
        self.dropout2 = nn.Dropout2d(p=trial.suggest_uniform("dropout_param_2", 0.1, 0.9))
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #features
        x = self.conv1(x)
        x = self.activation(x)
        x = self.maxpool_k3_s2(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.maxpool_k3_s2(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.conv5(x)
        x = self.activation(x)
        x = self.maxpool_k3_s2(x)
        #avgpool
        x = self.avgpool(x)
        #classifier
        x = torch.flatten(x, 1)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x


def train(model: nn.Module, device: str, train_loader: DataLoader, optimizer: optim.Optimizer) -> None:
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        print('Iteration: {} | Loss: {:1.5f}, '.format(batch_idx,loss.item()))
        loss.backward()
        optimizer.step()    
    


def test(model: nn.Module, device: str, test_loader: DataLoader) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    return 1 - correct / len(test_loader.dataset)


def get_optimizer(trial: optuna.trial.Trial, model: nn.Module) -> optim.Optimizer:

    def adam(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def momentum(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    optimizer_name = trial.suggest_categorical('optimizer', ['adam', 'momentum'])
    optimizer: Callable[[nn.Module, float, float], optim.Optimizer] = locals()[optimizer_name]
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    return optimizer(model, lr, weight_decay)


def objective_wrapper(train_loader: DataLoader, test_loader: DataLoader,
                      epochs: int) -> Callable[[optuna.trial.Trial], float]:

    def objective(trial: optuna.trial.Trial) -> float:
        model = AlexNet(trial=trial, num_classes=2).to(device)
        optimizer = get_optimizer(trial, model)

        for step in range(epochs):
            train(model, device, train_loader, optimizer)
            error_rate = test(model, device, test_loader)
            print('Epoch: {} | error_rate: {:1.5f}'.format(step+1, error_rate))
            trial.report(error_rate, step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return error_rate

    return objective


def main(epochs= 10, batchs= 128, trials= 100) -> None:
    output= 'result.csv'
    outJson= '/content/best_params.json'
    train_loader, test_loader = create_loader(batchs)
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    objective = objective_wrapper(train_loader, test_loader, epochs)
    study.optimize(objective, n_trials=trials)

    print(f"Best parameters: {study.best_params}")
    print(f"Best error rate: {study.best_value}")

    with open('./best_params.json', 'w') as outfile:
      json.dump(study.best_params, outfile)
    study.trials_dataframe().to_csv('result.csv')
    plot_optimization_history(study).show()