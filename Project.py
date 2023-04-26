import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tsne_torch import TorchTSNE
from gradcam import GradCAM, GradCAMpp
from gradcam.utils import visualize_cam



# Preprocessing: normalization and data augmentation
def get_transforms():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    return transform_train, transform_test

def display_image(image, title, name):
    plt.imshow(image.cpu().permute(1, 2, 0))
    plt.title(title)
    plt.show()
    plt.savefig(f"{name}_image.jpg")

def cutmix(data, targets, alpha=1.0):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets = (targets, shuffled_targets, lam)

    return data, targets


def cutmix_criterion(criterion, outputs, targets):
    targets1, targets2, lam = targets
    return lam * criterion(outputs, targets1) + (1 - lam) * criterion(outputs, targets2)


def mixup(inputs, targets, alpha):
    batch_size = inputs.size(0)
    indices = torch.randperm(batch_size)
    shuffled_inputs = inputs[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    mixed_inputs = lam * inputs + (1 - lam) * shuffled_inputs
    return mixed_inputs, (targets, shuffled_targets, lam)


def mixup_criterion(criterion, preds, targets):
    targets1, targets2, lam = targets
    return lam * criterion(preds, targets1) + (1 - lam) * criterion(preds, targets2)

# Plot training and validation curves
def plot_curves(train_losses, val_losses,name):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(f"{name}_curves.jpg")

# Plot t-SNE visualization
def plot_tsne(features, labels,name):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
    reduced_features = tsne.fit_transform(features)
    plt.figure()
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", 10), marker='o', s=20)
    plt.colorbar(ticks=range(10))
    plt.clim(-0.5, 9.5)
    plt.show()
    plt.savefig(f"{name}_plot_tsne.jpg")

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred,name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    plt.savefig(f"{name}_confusion.jpg")

def run_experiment(name,trainset, testset, data_proportion, num_epochs, use_supervised, use_mixup, use_cutmix, lambda_=0.1, alpha=1.0,unlabeled_proportion=0.1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=None, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_losses = []
    val_losses = []

    if use_supervised:
        train_size = int(train_test_split * len(trainset) * data_proportion)
        val_size = int((len(trainset) * data_proportion) - train_size)
        
#         print(f"train_size = {train_size}")
#         print(f"val_size = {val_size}")

        train_indices = list(range(0, train_size))
        train_subset = torch.utils.data.Subset(trainset, train_indices)
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=100, shuffle=True, num_workers=2)

        val_indices = list(range(train_size, train_size + val_size))
        val_subset = torch.utils.data.Subset(trainset, val_indices)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=100, shuffle=False, num_workers=2)
        
        test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


    else:
        
        # Semi-supervised learning
        labeled_size = int(train_test_split * len(trainset) * ( 1 - unlabeled_proportion) * data_proportion)
        unlabeled_size = int(train_test_split * len(trainset) * unlabeled_proportion * data_proportion)
        val_size = int((len(trainset) * data_proportion)  - labeled_size - unlabeled_size)
        
#         print(f"labeled_size = {labeled_size}")
#         print(f"unlabeled_size = {unlabeled_size}")
#         print(f"val_size = {val_size}")

        
        labeled_indices = list(range(0, labeled_size))
        labeled_subset = torch.utils.data.Subset(trainset, labeled_indices)
        labeled_loader = torch.utils.data.DataLoader(labeled_subset, batch_size=100, shuffle=True, num_workers=2)

        unlabeled_indices = list(range(labeled_size, labeled_size + unlabeled_size))
        unlabeled_subset = torch.utils.data.Subset(trainset, unlabeled_indices)
        unlabeled_loader = torch.utils.data.DataLoader(unlabeled_subset, batch_size=100, shuffle=True, num_workers=2)

        val_indices = list(range(labeled_size + unlabeled_size, labeled_size + unlabeled_size + val_size))
        val_subset = torch.utils.data.Subset(trainset, val_indices)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=100, shuffle=False, num_workers=2)

        test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    display_mixup = True
    display_cutmix = True
    for epoch in range(num_epochs):
        running_loss = 0.0
        if use_supervised:
            for i, (data, targets) in enumerate(train_loader):
                
                data, targets = data.to(device), targets.to(device)
                optimizer.zero_grad()

                outputs = model(data)
                loss = criterion(outputs, targets)

                if use_mixup:
                    mixup_data, mixup_targets = mixup(data.clone(), targets.clone(), alpha)
                    mixup_data = mixup_data.to(device)
                    mixup_outputs = model(mixup_data)
                    if display_mixup:
                        display_image(mixup_data[0], "MixUp Sample Image", f"{name}_mixup")
                        display_mixup = False
                    mixup_loss = mixup_criterion(criterion, mixup_outputs, mixup_targets)

                    loss += mixup_loss

                if use_cutmix:
                    cutmix_data, cutmix_targets = cutmix(data.clone(), targets.clone(), alpha)
                    cutmix_data = cutmix_data.to(device)
                    cutmix_outputs = model(cutmix_data)
                    if display_cutmix:
                        display_image(cutmix_data[0], "CutMix Sample Image", f"{name}_cutmix")
                        display_cutmix = False
                    cutmix_loss = cutmix_criterion(criterion, cutmix_outputs, cutmix_targets)

                    loss += cutmix_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_losses.append(running_loss / len(train_loader))

        else:
            # Semi-supervised learning
            for i, ((labeled_data, labeled_targets), (unlabeled_data, _)) in enumerate(zip(labeled_loader, unlabeled_loader)):
                labeled_data, labeled_targets = labeled_data.to(device), labeled_targets.to(device)
                optimizer.zero_grad()

                labeled_outputs = model(labeled_data)
                labeled_loss = criterion(labeled_outputs, labeled_targets)

                unlabeled_data = unlabeled_data.to(device)
                unlabeled_outputs = model(unlabeled_data)
                pseudo_labels = torch.argmax(unlabeled_outputs, dim=1)
                unlabeled_loss = criterion(unlabeled_outputs, pseudo_labels)

                loss = labeled_loss + lambda_ * unlabeled_loss

                if use_mixup:
                    mixup_data, mixup_targets = mixup(labeled_data.clone(), labeled_targets.clone(), alpha)
                    mixup_data = mixup_data.to(device)
                    mixup_outputs = model(mixup_data)
                    if display_mixup:
                        display_image(mixup_data[0], "MixUp Sample Image", f"{name}_mixup")
                        display_mixup = False
                    mixup_loss = mixup_criterion(criterion, mixup_outputs, mixup_targets)

                    loss += mixup_loss

                if use_cutmix:
                    cutmix_data, cutmix_targets = cutmix(labeled_data.clone(), labeled_targets.clone(), alpha)
                    cutmix_data = cutmix_data.to(device)
                    cutmix_outputs = model(cutmix_data)
                    if display_cutmix:
                        display_image(cutmix_data[0], "CutMix Sample Image", f"{name}_cutmix")
                        display_cutmix = False
                    cutmix_loss = cutmix_criterion(criterion, cutmix_outputs, cutmix_targets)

                    loss += cutmix_loss

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            train_losses.append(running_loss / len(labeled_loader))

        # Validation
        val_loss = 0.0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)

                outputs = model(data)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))
        
    # Plot training and validation curves
    plot_curves(train_losses, val_losses,name)

    # Calculate test accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    print(f'Test accuracy: {100 * correct / total:.2f}%')
    
    # Plot confusion matrix
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    plot_confusion_matrix(y_true, y_pred,name)

    return train_losses, val_losses



# Set your desired unlabeled_ratio and data_proportion values
unlabeled_ratio = 0.1
data_proportion = 0.2
train_test_split = 0.8
num_epochs = 50
lambda_ = 0.3
alpha = 0.3
ur=0.1

# Load data and run the experiment
transform_train, transform_test = get_transforms()
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

unsup = run_experiment("Baseline",trainset, testset, data_proportion, num_epochs, use_supervised=True, use_mixup=False, use_cutmix=False, lambda_ = lambda_, alpha = alpha, unlabeled_proportion = ur)
# unsup = run_experiment(trainset, testset, data_proportion, num_epochs, use_supervised=True, use_mixup=False, use_cutmix=True, lambda_ = lambda_, alpha = alpha, unlabeled_ratio = ur)
# unsup = run_experiment(trainset, testset, data_proportion, num_epochs, use_supervised=True, use_mixup=True, use_cutmix=False, lambda_ = lambda_, alpha = alpha, unlabeled_ratio = ur)
# unsup = run_experiment(trainset, testset, data_proportion, num_epochs, use_supervised=True, use_mixup=True, use_cutmix=True, lambda_ = lambda_, alpha = alpha, unlabeled_ratio = ur)
res = {}
for x in range(1,9,2):
    print(f"--------------{x*unlabeled_ratio *100}")
    ur = round((unlabeled_ratio * x),1)
    sub_result = []
    sub_result.append(run_experiment(f"EM_{ur}",trainset, testset, data_proportion, num_epochs, use_supervised=False, use_mixup=False, use_cutmix=False, lambda_ = lambda_, alpha = alpha, unlabeled_proportion = ur))
    sub_result.append(run_experiment(f"EM_CutMix_{ur}",trainset, testset, data_proportion, num_epochs, use_supervised=False, use_mixup=False, use_cutmix=True, lambda_ = lambda_, alpha = alpha, unlabeled_proportion = ur))
    sub_result.append(run_experiment(f"EM_MixUp_{ur}",trainset, testset, data_proportion, num_epochs, use_supervised=False, use_mixup=True, use_cutmix=False, lambda_ = lambda_, alpha = alpha, unlabeled_proportion = ur))
    sub_result.append(run_experiment(f"EM_CutMix_MixUp{ur}",trainset, testset, data_proportion, num_epochs, use_supervised=False, use_mixup=True, use_cutmix=True, lambda_ = lambda_, alpha = alpha, unlabeled_proportion = ur))
    res[ur] = sub_result