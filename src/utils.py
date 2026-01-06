from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

def get_loaders(dataset_name="MNIST", batch_size=64):
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data"

    if dataset_name == "CIFAR10":
        # RGB stats for CIFAR-10
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        data_class = datasets.CIFAR10
    else:
        # Grayscale stats for MNIST
        mean, std = (0.1307,), (0.3081,)
        data_class = datasets.MNIST

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_set = data_class(root=str(data_path), train=True, download=True, transform=transform)
    test_set = data_class(root=str(data_path), train=False, download=True, transform=transform)
    
    return DataLoader(train_set, batch_size=batch_size, shuffle=True), \
           DataLoader(test_set, batch_size=batch_size, shuffle=False)