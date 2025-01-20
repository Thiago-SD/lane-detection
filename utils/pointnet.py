import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


class PointCloudDataset(Dataset):
    def __init__(self, ply_dir, labels_file):
        self.ply_files = [os.path.join(ply_dir, f) for f in os.listdir(ply_dir) if f.endswith('.ply')]
        self.labels = self.load_labels(labels_file)

    def load_labels(self, labels_file):
        with open(labels_file, 'r') as f:
            lines = f.readlines()
        return {line.split(',')[0]: float(line.split(',')[1]) for line in lines}

    def __len__(self):
        return len(self.ply_files)

    def __getitem__(self, idx):
        ply_file = self.ply_files[idx]
        file_name = os.path.basename(ply_file)
        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points, dtype=np.float32)
        label = self.labels[file_name]
        return torch.tensor(points, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
    
class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, N, C) -> (B, C, N)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.max(x, 2)[0]  # Global max pooling
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(train_loader, model, epochs=10, lr=0.001, save_path="pointnet_model.pth"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for points, label in train_loader:
            optimizer.zero_grad()
            output = model(points)
            loss = criterion(output.squeeze(), label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Modelo salvo em: {save_path}")

def test_model(test_loader, model, load_path="pointnet_model.pth"):
    model.load_state_dict(torch.load(load_path))
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for points, label in test_loader:
            prediction = model(points).item()
            predictions.append(prediction)
            labels.append(label.item())
    return predictions, labels

def main():
    # Diretórios
    script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_path, "../data")
    train_dir = data_path+"/pointclouds/ROI"
    test_dir = data_path+"/pointclouds/ROI"
    labels_train = "caminho/para/labels_treino.csv"  # Arquivo no formato: "arquivo.ply,distância"
    labels_test = "caminho/para/labels_teste.csv"
    x_limits = (-5, 5)  # Largura lateral
    y_limits = (0, 30)  # Profundidade frontal

    # Carregar datasets
    train_dataset = PointCloudDataset(train_dir, labels_train, x_limits, y_limits)
    test_dataset = PointCloudDataset(test_dir, labels_test, x_limits, y_limits)

    # Configurar DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Modelo
    model = PointNet()

    if not os.path.exists(script_path+"/model"):
        os.makedirs(script_path+"/model")

    # Treinar o modelo
    train_model(train_loader, model, epochs=10, save_path="/model/lane_detection_model.pth")

    # Testar o modelo
    predictions, labels = test_model(test_loader, model, load_path="/model/lane_detection_model.pth")

    # Visualizar resultados
    plt.plot(labels, predictions, 'o', label="Predições")
    plt.plot(labels, labels, 'r', label="Ideal")
    plt.xlabel("Rótulos Reais")
    plt.ylabel("Predições")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()