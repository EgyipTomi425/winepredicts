import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import deque
import random
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wine_data = pd.read_csv("WineQT.csv")
wine_data = wine_data.drop(columns=['Id'])

# Jellemzők és célváltozó elkülönítése
X = wine_data.drop('quality', axis=1)
y = wine_data['quality']

# Adatok felosztása tanító és teszt készletre
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class WineQualityNet(nn.Module):
    def __init__(self, dropout_rate=0.125):
        super(WineQualityNet, self).__init__()
        self.fc1 = nn.Linear(11, 1024)  # 11 jellemző
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(1024, 1)  # 1 kimenet (quality)

        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.015625)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.015625)

    def forward(self, x):
        x = self.leaky_relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.leaky_relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=512, patience=16):
    best_loss = float('inf')
    best_epoch = -1
    best_train_loss = None
    best_model_wts = model.state_dict()
    loss_history = deque(maxlen=patience)  # Az utolsó N epoch teszt veszteségének tárolása, ahol N a türelmi limit

    for epoch in range(num_epochs):
        model.train()  # Tanítási mód bekapcsolása
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)

        # Tesztelési veszteség számítása
        model.eval()  # Értékelési mód bekapcsolása
        test_loss = 0.0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
        
        test_loss /= len(test_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        # Korai leállítás logikája és a legjobb modell mentése
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            best_train_loss = train_loss
            best_model_wts = model.state_dict()
            loss_history.clear()
        else:
            loss_history.append(test_loss)
            # Ha 'patience' számú epochon keresztül nem javul a teszt veszteség
            if len(loss_history) == patience and test_loss > min(loss_history):
                print(f"\nEarly stopping triggered after {patience} epochs at epoch {best_epoch + 1}. Restoring best model weights.")
                print(f"Best Train Loss: {best_train_loss:.4f}, Best Test Loss: {best_loss:.4f}")
                model.load_state_dict(best_model_wts)
                break

    return model

def simulated_annealing(model, X_min, X_max, max_iterations=1024, initial_temperature=8.0):
    best_quality = float('-inf')
    best_input = None
    input_size = X_min.shape[0]
    current_input = torch.rand(input_size, device=device) * (X_max - X_min) + X_min  # Jelenlegi állapot
    best_input = current_input.clone()
    temperature = initial_temperature

    for _ in range(max_iterations):
        # Véletlenszerű módosítás az aktuális bemeneten
        new_input = current_input + torch.randn(input_size, device=device) * temperature
        new_input = torch.max(torch.min(new_input, X_max), X_min)  # Korlátok betartása

        current_output = model(current_input.unsqueeze(0))
        new_output = model(new_input.unsqueeze(0))

        current_quality = current_output.item()
        new_quality = new_output.item()

        # Elfogadjuk-e az új állapotot?
        if new_quality > current_quality or torch.rand(1).item() < torch.exp(torch.tensor((new_quality - current_quality) / temperature)):
            current_input = new_input.clone()
            current_quality = new_quality

        if current_quality > best_quality:
            best_quality = current_quality
            best_input = current_input.clone()

        if(_%(1024/8)==0):
            current_input_numpy = current_input.cpu().detach().numpy()  # Konvertálás NumPy tömbbé
            new_input_numpy = new_input.cpu().detach().numpy()  # Konvertálás NumPy tömbbé
            print(f"Iteration: {_}, Temp: {round(temperature,4)}, Current Input: {[round(val, 2) for val in current_input_numpy]}, Pred: {round(current_output.item(), 2)}, New Input: {[round(val, 2) for val in new_input_numpy]}, Pred: {round(new_output.item(), 2)}")

        temperature*=.9921875

    return best_input

model = WineQualityNet(dropout_rate=0.125).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0078125)

trained_model = train_model(model, criterion, optimizer, train_loader, test_loader)

total_params = sum(p.numel() for p in model.parameters())
print(f'Total number of parameters: {total_params},\n')

# Szimulált hűtés maximumkereséssel
X_min = torch.tensor(X_train.min().values, dtype=torch.float32, device=device)
X_max = torch.tensor(X_train.max().values, dtype=torch.float32, device=device)
best_input = simulated_annealing(model, X_min, X_max)

column_names = X.columns.tolist()
name_value_pairs = [(column_names[i], best_input[i].item()) for i in range(len(column_names))]

# X_max és X_min kiíratása
print("\nAttribute: MIN - MAX")
for name, max_value, min_value in zip(column_names, X_max, X_min):
    print(f"{name}: {min_value.item():.2f} - {max_value.item():.2f}")

# Név-érték párok kiíratása
print("\nBest Input Parameters Found:")
for name, value in name_value_pairs:
    print(f"{name}: {value:.2f}")

# Prediktált minőség kiíratása
predicted_quality = model(best_input.unsqueeze(0)).item()
print("Predicted Quality:", predicted_quality)