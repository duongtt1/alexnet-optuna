from typing import Any, Callable, List, Tuple
import optuna  # type: ignore
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision  # type: ignore
from torchvision.datasets import ImageFolder , MNIST # type: ignore
import json

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

best_params_file = open('./best_params.json','r')
congfig = json.load(best_params_file)

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_loader(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    input_size = 224

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
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
    def __init__(self, cfg, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=cfg['params_dropout_prob1']),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=cfg['params_dropout_prob2']),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        output = F.log_softmax(x, dim=1)
        return output

def alexnet(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> AlexNet:
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

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


        
def main(epochs= 10, batchs= 128) -> None:
    train_loader, test_loader = create_loader(batchs)
    model = AlexNet(congfig, num_classes=2).to(device)
    if congfig['params_optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=congfig['params_lr'], weight_decay=congfig['params_weight_decay'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=congfig['params_lr'], momentum=0.9, weight_decay=congfig['params_weight_decay'])
    for step in range(epochs):
        train(model, device, train_loader, optimizer,step)
        error_rate = test(model, device, test_loader)
        print('Epoch: {} | error_rate: {:1.5f}'.format(step+1, error_rate))
