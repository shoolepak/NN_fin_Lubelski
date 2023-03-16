import warnings
warnings.filterwarnings("ignore", message="No audio backend is available")  # this should make it so it does not try
#  to play the sound
import torch
import os
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as TV
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

img_path = os.path.join(str(Path(__file__).parent), "recordings")

#   Define the neural network architecture/
class SpokenDigitRecognizerr(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SpokenDigitRecognizerr, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


#   use gpu when possible if not use cpu(duh)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#   actually, kinda useless to have it all here, but whatever
learning_rate= 0.001
batch_size= 64
num_classes= 10
num_epochs= 10


class LocalSpokenDigitDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transforms = TV.Compose([T.MelSpectrogram(sample_rate=8000),
                                     T.FrequencyMasking(freq_mask_param=10),
                                     T.TimeMasking(time_mask_param=10),
                                     TV.Resize((32, 32)),
                                     TV.ToTensor()])
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for filename in os.listdir(class_dir):
                path = os.path.join(class_dir, filename)
                self.samples.append((path, self.class_to_idx[class_name]))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        waveform, sample_rate = torchaudio.load(path)
        spectrogram = self.transforms(waveform)
        return spectrogram, label


dataset = LocalSpokenDigitDataset("recordings")

train_data, test_data = train_test_split(dataset, test_size=0.2)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

#   ah yes, num_classes=num_classes
model = SpokenDigitRecognizerr(input_size=13, hidden_size=64, num_layers=2, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#   train dis bad bois train set
def train(model, dataloader, optimizer, criterion):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        preds = torch.argmax(output, dim=1)
        train_acc += accuracy_score(target.cpu().numpy(), preds.cpu().numpy()) * data.size(0)
    train_loss /= len(dataloader.dataset)
    train_acc /= len(dataloader.dataset)
    return train_loss, train_acc

#   test
def test(model, dataloader, criterion):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            preds = torch.argmax(output, dim=1)
            test_acc += accuracy_score(target.cpu().numpy(), preds.cpu().numpy()) * data.size(0)
    test_loss /= len(dataloader.dataset)
    test_acc /= len(dataloader.dataset)
    return test_loss, test_acc

#   Train the model
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    test_loss, test_acc = test(model, test_loader, criterion)
    print('Epoch {}/{}: Train Loss: {:.6f}, Train Acc: {:.6f}, Test Loss: {:.6f}, Test Acc: {:.6f}'.format(epoch+1, num_epochs, train_loss, train_acc, test_loss, test_acc))


