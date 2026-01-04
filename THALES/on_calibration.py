import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

# --- 1. Data Loading & Preprocessing ---
print("Loading CIFAR-100 data (this may take a moment)...")
batch_size = 128

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR100(root='THALES/data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR100(root='THALES/data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- 2. Model Definitions ---

class LeNet5(nn.Module):
    def __init__(self, num_classes=100):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNetCIFAR, self).__init__()
        self.in_planes = 32
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.linear = nn.Linear(128*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# --- 3. Training & Evaluation Helper ---

def train_model(model, train_loader, epochs=5, lr=0.01, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=4e-3)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
    return model

def get_predictions(model, test_loader, device='cpu'):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    confidences = np.max(all_probs, axis=1)
    predictions = np.argmax(all_probs, axis=1)
    accuracy_bool = (predictions == all_labels)
    
    return confidences, accuracy_bool

# --- 4. Plotting Functions (Same as before) ---
def compute_calibration_metrics(predictions, truth, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    ece = 0.0
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy = truth[in_bin].mean()
            avg_confidence = predictions[in_bin].mean()
            accuracies.append(accuracy)
            confidences.append(avg_confidence)
            ece += np.abs(avg_confidence - accuracy) * prop_in_bin
        else:
            accuracies.append(0)
            confidences.append(0)
            
    return np.array(accuracies), np.array(confidences), bin_boundaries, ece

def plot_reliability_diagram(ax, accuracies, confidences, bin_boundaries, ece, title="Reliability Diagram"):
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    width = np.diff(bin_boundaries)[0]
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.bar(bin_centers, accuracies, width=width, edgecolor='black', color='blue', alpha=0.8, label='Outputs')
    for i in range(len(accuracies)):
        acc = accuracies[i]
        conf = confidences[i]
        if conf > acc and conf > 0:
            ax.bar(bin_centers[i], conf - acc, bottom=acc, width=width, 
                   edgecolor='red', color='pink', hatch='//', alpha=0.5, label='Gap' if i == 0 else "")
    
    ax.set_title(title)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper left')
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.5, 0.2, f"ECE = {ece*100:.2f}%", transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=props)

def plot_confidence_histogram(ax, predictions, title="Confidence Histogram"):
    ax.hist(predictions, bins=15, range=(0,1), edgecolor='black', color='blue', alpha=0.8)
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 1)

# --- 5. Main ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Init Models
lenet = LeNet5(num_classes=100)
# Small ResNet: Block, [2, 2, 2] (ResNet18 structure but smaller width potentially)
resnet = ResNetCIFAR(BasicBlock, [2, 2, 2], num_classes=100) 

# Train
print("Training LeNet on CIFAR-100...")
lenet = train_model(lenet, train_loader, epochs=25, device=device)
print("Training ResNet on CIFAR-100...")
resnet = train_model(resnet, train_loader, epochs=15, device=device)

# Evaluate
print("Evaluating...")
lenet_conf, lenet_acc = get_predictions(lenet, test_loader, device=device)
resnet_conf, resnet_acc = get_predictions(resnet, test_loader, device=device)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# LeNet
plot_confidence_histogram(axes[0, 0], lenet_conf, title=f"LeNet (CIFAR-100)\nAccuracy: {lenet_acc.mean():.2%}")
acc, conf, bins, ece = compute_calibration_metrics(lenet_conf, lenet_acc)
plot_reliability_diagram(axes[1, 0], acc, conf, bins, ece, title="LeNet Reliability")

# ResNet
plot_confidence_histogram(axes[0, 1], resnet_conf, title=f"ResNet (CIFAR-100)\nAccuracy: {resnet_acc.mean():.2%}")
acc, conf, bins, ece = compute_calibration_metrics(resnet_conf, resnet_acc)
plot_reliability_diagram(axes[1, 1], acc, conf, bins, ece, title="ResNet Reliability")

plt.tight_layout()
plt.show()
