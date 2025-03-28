import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

NUM_EPOCHS = 100

class LaneDataset(Dataset):
    def __init__(self, data_path, normalize_params=None):
        print(f"Carregando dataset de {data_path}...")
        data = np.load(data_path, allow_pickle=True)
        self.pointclouds = data['pointclouds']
        self.distances = data['distances']
        
        if normalize_params:
            self.normalize_params = normalize_params
        else:
            print("Calculando parâmetros de normalização...")
            self.normalize_params = {
                'distance': {
                    'mean': np.mean(self.distances),
                    'std': np.std(self.distances)
                }
            }
        print(f"Dataset carregado com {len(self)} amostras.")
    
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
            'coordinates': torch.FloatTensor(coords),
            'distance': torch.FloatTensor([distance])
        }

def split_dataset(dataset, test_size=0.2):
    print(f"Dividindo dataset (treino: {1-test_size:.0%}, teste: {test_size:.0%})...")
    test_len = int(len(dataset) * test_size)
    train_len = len(dataset) - test_len
    return random_split(dataset, [train_len, test_len])

class PointNetPP(nn.Module):
    def __init__(self, norm_mean=0.0, norm_std=1.0):
        super().__init__()
        print("Inicializando modelo PointNet++...")
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
        print("Modelo inicializado com sucesso.")
    
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
    # Inicialização
    full_dataset = LaneDataset(data_path)
    train_dataset, test_dataset = split_dataset(full_dataset, test_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = PointNetPP(
        norm_mean=full_dataset.normalize_params['distance']['mean'],
        norm_std=full_dataset.normalize_params['distance']['std']
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Armazenamento de métricas
    train_losses = []
    test_losses = []
    r2_scores = []
    mae_scores = []
    
    print("\nIniciando treinamento...")
    for epoch in range(epochs):
        # Treino
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            coords, target = batch['coordinates'], batch['distance']
            coords = coords.view(len(target), -1, 3)
            
            optimizer.zero_grad()
            output = model(coords)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Avaliação
        model.eval()
        epoch_test_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                coords, target = batch['coordinates'], batch['distance']
                coords = coords.view(len(target), -1, 3)
                output = model(coords)
                
                # Armazena para cálculo de métricas
                all_preds.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
                
                epoch_test_loss += criterion(output, target).item()
        
        # Cálculo de métricas
        epoch_train_loss /= len(train_loader)
        epoch_test_loss /= len(test_loader)
        
        # Desnormaliza para cálculo das métricas
        all_targets = np.array(all_targets) * full_dataset.normalize_params['distance']['std'] + full_dataset.normalize_params['distance']['mean']
        all_preds = np.array(all_preds) * full_dataset.normalize_params['distance']['std'] + full_dataset.normalize_params['distance']['mean']
        
        r2 = r2_score(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        
        # Armazena métricas
        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)
        r2_scores.append(r2)
        mae_scores.append(mae)
        
        # Log de progresso
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}")
        print(f"  Test Loss: {epoch_test_loss:.4f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        # Plotagem periódica
        if (epoch+1) % 10 == 0 or epoch == epochs-1:
            plot_metrics(train_losses, test_losses, r2_scores, mae_scores, all_preds, all_targets)
    
    return model, test_dataset, (train_losses, test_losses, r2_scores, mae_scores)

def plot_metrics(train_losses, test_losses, r2_scores, mae_scores, all_preds, all_targets):
    plt.figure(figsize=(15, 10))
    
    # Subplot para perdas
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Subplot para R² Score
    plt.subplot(2, 2, 2)
    plt.plot(r2_scores, label='R² Score', color='green')
    plt.title('R² Score Evolution')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.legend()
    plt.grid(True)
    
    # Subplot para MAE
    plt.subplot(2, 2, 3)
    plt.plot(mae_scores, label='MAE', color='red')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    # Subplot para distribuição de erros
    plt.subplot(2, 2, 4)
    errors = [pred - true for true, pred in zip(all_targets, all_preds)]
    plt.hist(errors, bins=30, alpha=0.7)
    plt.title('Error Distribution')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    print("Gráficos de métricas salvos em 'training_metrics.png'")

def evaluate_model(model, test_dataset):
    print("\nAvaliando modelo no conjunto de teste...")
    model.eval()
    all_preds = []
    all_targets = []
    
    for i in range(len(test_dataset)):
        sample = test_dataset[i]
        coords = sample['coordinates'].unsqueeze(0)
        pred = model(coords, denormalize=True).item()
        true = sample['distance'].item() * test_dataset.dataset.normalize_params['distance']['std'] + test_dataset.dataset.normalize_params['distance']['mean']
        
        all_preds.append(pred)
        all_targets.append(true)
    
    # Cálculo de métricas
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    
    print("\n=== Métricas Finais ===")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot de resultados
    plt.figure(figsize=(10, 5))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
    plt.xlabel('Valores Reais')
    plt.ylabel('Predições')
    plt.title('Predições vs Valores Reais')
    plt.grid(True)
    plt.savefig('predictions_vs_actuals.png')
    plt.close()
    print("Gráfico de predições vs reais salvo em 'predictions_vs_actuals.png'")

def predict_distance(model, pointcloud):
    print("\nFazendo predição para nova pointcloud...")
    if len(pointcloud) > 1000:
        pointcloud = pointcloud[np.random.choice(len(pointcloud), 1000, replace=False)]
    
    coords = pointcloud[:, :3]
    coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
    coords = torch.FloatTensor(coords).unsqueeze(0)
    
    with torch.no_grad():
        model.eval()
        pred = model(coords, denormalize=True)
    
    print(f"Predição concluída: {pred.item():.2f}")
    return pred.item()

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "training_data", "complete_training_data.npz")
    
    try:
        # Treinar e avaliar
        model, test_dataset, metrics = train_model(data_path, epochs=NUM_EPOCHS, batch_size=64)
        torch.save(model.state_dict(), "lane_distance_regressor.pth")
        print("Modelo treinado e salvo com sucesso.")
        
        # Avaliação detalhada
        evaluate_model(model, test_dataset)
        
        # Exemplo de predição
        sample_data = np.load(data_path, allow_pickle=True)
        sample_pc = sample_data['pointclouds'][0]
        distance = predict_distance(model, sample_pc)
        print(f"\nExemplo de Predição: {distance:.2f} m")
        
    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")
        raise