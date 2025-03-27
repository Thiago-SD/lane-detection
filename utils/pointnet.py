import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class LaneDataset(Dataset):
    def __init__(self, data_path, normalize_params=None):
        data = np.load(data_path, allow_pickle=True)
        self.pointclouds = data['pointclouds']
        self.distances = data['distances']
        
        # Usa parâmetros de normalização fornecidos ou calcula novos
        if normalize_params:
            self.normalize_params = normalize_params
        else:
            self.normalize_params = {
                'distance': {
                    'mean': np.mean(self.distances),
                    'std': np.std(self.distances)
                }
            }
    
    def __len__(self):
        return len(self.distances)
    
    def __getitem__(self, idx):
        pc = self.pointclouds[idx]
        
        if len(pc) > 1000:
            pc = pc[np.random.choice(len(pc), 1000, replace=False)]
        
        coords = pc[:, :3]
        coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
        
        distance = (self.distances[idx] - self.normalize_params['distance']['mean']) / \
                   self.normalize_params['distance']['std']
        
        return {
            'coordinates': torch.FloatTensor(coords),  # Shape: (1000, 3)
            'distance': torch.FloatTensor([distance])   # Shape: (1,)
        }

def split_dataset(dataset, test_size=0.2):
    """Divide o dataset em treino e teste de forma aleatória"""
    test_len = int(len(dataset) * test_size)
    train_len = len(dataset) - test_len
    return random_split(dataset, [train_len, test_len])

class PointNetPP(nn.Module):
    def __init__(self, norm_mean=0.0, norm_std=1.0):
        super().__init__()
        self.register_buffer('norm_mean', torch.tensor(norm_mean))
        self.register_buffer('norm_std', torch.tensor(norm_std))
        
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x, denormalize=False):
        batch_size, num_points, _ = x.shape
        x = x.view(-1, 3)
        x = self.mlp1(x)
        x = x.view(batch_size, num_points, -1)
        x = torch.max(x, dim=1)[0]
        x = self.mlp2(x)
        
        if denormalize:
            return x * self.norm_std + self.norm_mean
        return x

def train_model(data_path, epochs=50, batch_size=32, test_size=0.2):
    # Carrega dataset completo
    full_dataset = LaneDataset(data_path)
    
    # Divide em treino e teste
    train_dataset, test_dataset = split_dataset(full_dataset, test_size)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Modelo com normalização baseada no treino
    model = PointNetPP(
        norm_mean=full_dataset.normalize_params['distance']['mean'],
        norm_std=full_dataset.normalize_params['distance']['std']
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Treinamento
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            coords, target = batch['coordinates'], batch['distance']
            coords = coords.view(len(target), -1, 3)
            
            output = model(coords)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Avaliação no teste
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                coords, target = batch['coordinates'], batch['distance']
                coords = coords.view(len(target), -1, 3)
                output = model(coords)
                test_loss += criterion(output, target).item()
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}")
    
    return model, test_dataset

def evaluate_model(model, test_dataset):
    model.eval()
    errors = []
    
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        coords = sample['coordinates'].unsqueeze(0)
        pred = model(coords, denormalize=True).item()
        true = sample['distance'].item() * test_dataset.dataset.normalize_params['distance']['std'] + test_dataset.dataset.normalize_params['distance']['mean']
        errors.append(abs(pred - true))
    
    print(f"\nAvaliação Final:")
    print(f"Erro Médio Absoluto: {np.mean(errors):.2f} m")
    print(f"Desvio Padrão do Erro: {np.std(errors):.2f} m")
    print(f"Erro Máximo: {np.max(errors):.2f} m")

def predict_distance(model, pointcloud):
    if len(pointcloud) > 1000:
        pointcloud = pointcloud[np.random.choice(len(pointcloud), 1000, replace=False)]
    
    coords = pointcloud[:, :3]
    coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
    coords = torch.FloatTensor(coords).unsqueeze(0)  # Shape: (1, 1000, 3)
    
    with torch.no_grad():
        model.eval()
        pred = model(coords, denormalize=True)
    
    return pred.item()

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "training_data", "complete_training_data.npz")
    
    # Treinar e avaliar
    model, test_dataset = train_model(data_path, epochs=2000, batch_size=64)
    torch.save(model.state_dict(), "lane_distance_regressor.pth")
    
    # Avaliação detalhada
    evaluate_model(model, test_dataset)
    
    # Exemplo de predição
    sample_data = np.load(data_path, allow_pickle=True)
    sample_pc = sample_data['pointclouds'][0]
    distance = predict_distance(model, sample_pc)
    print(f"\nExemplo de Predição: {distance:.2f} m")