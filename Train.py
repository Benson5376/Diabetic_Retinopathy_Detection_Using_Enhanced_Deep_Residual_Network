import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim
import numpy as np
import torchvision.models as torch_models
from dataloader import RetinopathyLoader, getData
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

train_data = RetinopathyLoader('new_train', 'train')
test_data = RetinopathyLoader('new_test', 'test')
train_data = DataLoader(train_data, batch_size=8, shuffle=True)
test_data = DataLoader(test_data, batch_size=8, shuffle=False)

class Pretrained_ResNet18(nn.Module):
    def __init__(self):
        
        super(Pretrained_ResNet18,self).__init__()
        
        # ResNet model from torchvision 
        self.model = models.resnet18(pretrained=True)
        
        # The number of output neuron of the last fc layer        
        
        num_features = self.model.fc.in_features
        
        # Replace the fc layer
        self.model.fc = nn.Linear(num_features, 5)
        
        
    def forward(self, x):
        x = self.model(x)
        return x

class ResNet18(nn.Module):
    def __init__(self):
        
        super(Pretrained_ResNet18,self).__init__()
        
        # ResNet model from torchvision 
        self.model = models.resnet18(pretrained=False)
        
        # The number of output neuron of the last fc layer        
        
        num_features = self.model.fc.in_features
        
        # Replace the fc layer
        self.model.fc = nn.Linear(num_features, 5)
        
        
    def forward(self, x):
        x = self.model(x)
        return x

class Pretrained_ResNet50(nn.Module):
    def __init__(self):
        
        super(Pretrained_ResNet18,self).__init__()
        
        # ResNet model from torchvision 
        self.model = models.resnet50(pretrained=True)
        
        # The number of output neuron of the last fc layer        
        
        num_features = self.model.fc.in_features
        
        # Replace the fc layer
        self.model.fc = nn.Linear(num_features, 5)
        
        
    def forward(self, x):
        x = self.model(x)
        return x

class ResNet50(nn.Module):
    def __init__(self):
        
        super(Pretrained_ResNet18,self).__init__()
        
        # ResNet model from torchvision 
        self.model = models.resnet50(pretrained=False)
        
        # The number of output neuron of the last fc layer        
        
        num_features = self.model.fc.in_features
        
        # Replace the fc layer
        self.model.fc = nn.Linear(num_features, 5)
        
        
    def forward(self, x):
        x = self.model(x)
        return x
    
# --------------------------------------ResNet18(Pretrained)---------------------------------------------
model = Pretrained_ResNet18()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
loss_function = torch.nn.CrossEntropyLoss()

train_acc = []
test_acc = []

print("ResNet18(Pretrained)")
for epoch in range(10):
    
    print("---------{}th epoch---------".format(epoch+1))
    
    model.train()
    train_loss = 0.0
    correct = 0
    
    for i, (data, label) in enumerate(train_data):
        
        label = label.type(torch.LongTensor)
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        
        correct += (output.argmax(dim=1) == label).sum().item()
        train_loss += loss.item()
    
    
    print("Accuracy: ")
    print(correct/28099)
    print("Training Loss: ")
    print(train_loss)
    train_acc.append(correct/28099)
    
    model.eval()
    with torch.no_grad():
        
        correct = 0
        
        for i, (data, label) in enumerate(test_data):
            
            label = label.type(torch.LongTensor)
            data = data.to(device)
            label = label.to(device)
            
            output = model(data)
            
            correct += (output.argmax(dim=1) == label).sum().item()
        
        print("Testing Accuracy: ")
        print(correct/7025)
        test_acc.append(correct/7025)
        
        if((correct/7025)>=0.82):
            torch.save(model, 'best_model.pt')


# --------------------------------------ResNet18(Without Pretrained)---------------------------------------------
model = ResNet18()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
loss_function = torch.nn.CrossEntropyLoss()

train_acc = []
test_acc = []

print("ResNet18(Without Pretrained)")
for epoch in range(10):
    
    print("---------{}th epoch---------".format(epoch+1))
    
    model.train()
    train_loss = 0.0
    correct = 0
    
    for i, (data, label) in enumerate(train_data):
        
        label = label.type(torch.LongTensor)
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        
        correct += (output.argmax(dim=1) == label).sum().item()
        train_loss += loss.item()
    
    
    print("Accuracy: ")
    print(correct/28099)
    print("Training Loss: ")
    print(train_loss)
    train_acc.append(correct/28099)
    
    model.eval()
    with torch.no_grad():
        
        correct = 0
        
        for i, (data, label) in enumerate(test_data):
            
            label = label.type(torch.LongTensor)
            data = data.to(device)
            label = label.to(device)
            
            output = model(data)
            
            correct += (output.argmax(dim=1) == label).sum().item()
        
        print("Testing Accuracy: ")
        print(correct/7025)
        test_acc.append(correct/7025)
        
        if((correct/7025)>=0.82):
            torch.save(model, 'best_model.pt')

# --------------------------------------ResNet50(Pretrained)---------------------------------------------
model = Pretrained_ResNet50()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
loss_function = torch.nn.CrossEntropyLoss()

train_acc = []
test_acc = []

print("ResNet50(Pretrained)")
for epoch in range(5):
    
    print("---------{}th epoch---------".format(epoch+1))
    
    model.train()
    train_loss = 0.0
    correct = 0
    
    for i, (data, label) in enumerate(train_data):
        
        label = label.type(torch.LongTensor)
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        
        correct += (output.argmax(dim=1) == label).sum().item()
        train_loss += loss.item()
    
    
    print("Accuracy: ")
    print(correct/28099)
    print("Training Loss: ")
    print(train_loss)
    train_acc.append(correct/28099)
    
    model.eval()
    with torch.no_grad():
        
        correct = 0
        
        for i, (data, label) in enumerate(test_data):
            
            label = label.type(torch.LongTensor)
            data = data.to(device)
            label = label.to(device)
            
            output = model(data)
            
            correct += (output.argmax(dim=1) == label).sum().item()
        
        print("Testing Accuracy: ")
        print(correct/7025)
        test_acc.append(correct/7025)
        
        if((correct/7025)>=0.82):
            torch.save(model, 'best_model.pt')

# --------------------------------------ResNet50(Without Pretrained)---------------------------------------------
model = ResNet50()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
loss_function = torch.nn.CrossEntropyLoss()

train_acc = []
test_acc = []

print("ResNet50(Without Pretrained)")
for epoch in range(5):
    
    print("---------{}th epoch---------".format(epoch+1))
    
    model.train()
    train_loss = 0.0
    correct = 0
    
    for i, (data, label) in enumerate(train_data):
        
        label = label.type(torch.LongTensor)
        data = data.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        
        correct += (output.argmax(dim=1) == label).sum().item()
        train_loss += loss.item()
    
    
    print("Accuracy: ")
    print(correct/28099)
    print("Training Loss: ")
    print(train_loss)
    train_acc.append(correct/28099)
    
    model.eval()
    with torch.no_grad():
        
        correct = 0
        
        for i, (data, label) in enumerate(test_data):
            
            label = label.type(torch.LongTensor)
            data = data.to(device)
            label = label.to(device)
            
            output = model(data)
            
            correct += (output.argmax(dim=1) == label).sum().item()
        
        print("Testing Accuracy: ")
        print(correct/7025)
        test_acc.append(correct/7025)
        
        if((correct/7025)>=0.82):
            torch.save(model, 'best_model.pt')