import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

NUM_EPOCHS = 3000
NUM_POINTS = 5000

class LaneDataset(Dataset):
    def __init__(self, npz_file, mode='train', num_points=1000):
        data = np.load(npz_file, allow_pickle=True)
        
        # Carrega os dados de treino ou teste
        if mode == 'train':
            self.pointclouds = data['train'].item()['pointclouds']
            self.distances = data['train'].item()['distances']
            self.positions = data['train'].item()['positions']
        else:
            self.pointclouds = data['test'].item()['pointclouds']
            self.distances = data['test'].item()['distances']
            self.positions = data['test'].item()['positions']
        
        # Calcula parâmetros de normalização
        self.normalize_params = {
            'distance': {
                'mean': np.mean(self.distances),
                'std': np.std(self.distances)
            }
        }
        
        self.num_points = num_points
        print(f"Dataset {mode} carregado com {len(self)} amostras")

    def __len__(self):
        return len(self.distances)

    def __getitem__(self, idx):
        pc = self.pointclouds[idx]
        
        # Amostragem de pontos (se necessário)
        if len(pc) > self.num_points:
            pc = pc[np.random.choice(len(pc), self.num_points, replace=False)]
        
        # Normalização das coordenadas
        coords = pc[:, :3]  # Pega apenas x,y,z (ignora intensidade se houver)
        coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
        
        # Normalização da distância
        distance = (self.distances[idx] - self.normalize_params['distance']['mean']) / \
                   self.normalize_params['distance']['std']
        
        return {
            'coordinates': torch.FloatTensor(coords),
            'distance': torch.FloatTensor([distance])
        }

class PointNetPP(nn.Module):
    def __init__(self, norm_mean=0.0, norm_std=1.0):
        super().__init__()
        print("Inicializando modelo PointNet++...")
        self.register_buffer('norm_mean', torch.tensor(norm_mean))
        self.register_buffer('norm_std', torch.tensor(norm_std))
        
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
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
    

def save_checkpoint(epoch, model, optimizer, train_losses, test_losses, r2_scores, mae_scores, checkpoint_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'r2_scores': r2_scores,
        'mae_scores': mae_scores
    }
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_pointnet.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint salvo em {checkpoint_dir}")

def load_latest_checkpoint(checkpoint_dir, model, optimizer):

    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_pointnet.pth')
    
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint carregado: {checkpoint_path} (época {checkpoint['epoch'] + 1})")
    
    return (
        checkpoint['epoch'],
        checkpoint['train_losses'],
        checkpoint['test_losses'],
        checkpoint['r2_scores'],
        checkpoint['mae_scores']
    )

def train_model(data_path, epochs=50, batch_size=32, num_points=1000, model_dir=None):
    # Carrega o dataset completo para normalização
    full_data = np.load(data_path, allow_pickle=True)
    train_data = full_data['train'].item()
    test_data = full_data['test'].item()
    
    # Calcula parâmetros de normalização globais
    normalize_params = {
        'distance': {
            'mean': np.mean(np.concatenate([train_data['distances'], test_data['distances']])),
            'std': np.std(np.concatenate([train_data['distances'], test_data['distances']]))
        }
    }
    
    # Cria os datasets
    train_dataset = LaneDataset(data_path, mode='train', num_points=num_points)
    test_dataset = LaneDataset(data_path, mode='test', num_points=num_points)
    
    # Sobrescreve os parâmetros de normalização com os globais
    train_dataset.normalize_params = normalize_params
    test_dataset.normalize_params = normalize_params
    
    # Verifica se há dados de teste
    if len(test_dataset) == 0:
        raise ValueError("Nenhum dado de teste encontrado.")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Inicializa o modelo
    model = PointNetPP(
        norm_mean=normalize_params['distance']['mean'],
        norm_std=normalize_params['distance']['std']
    )
    
    # Restante da função permanece igual...
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []
    r2_scores = []
    mae_scores = []

    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    start_epoch = 0
    train_losses, test_losses, r2_scores, mae_scores = [], [], [], []

    if os.path.exists(os.path.join(checkpoint_dir, 'checkpoint_pointnet.pth')):
        start_epoch, train_losses, test_losses, r2_scores, mae_scores = load_latest_checkpoint(checkpoint_dir, model, optimizer)
        print(f"Retomando treinamento da época {start_epoch + 1}:")

    print("Iniciando treinamento...")
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_train_loss = 0
        for batch in train_loader:
            coords, target = batch['coordinates'], batch['distance']
            coords = coords.view(len(target), -1, 3)  # Reshape para (batch_size, num_points, 3)
            
            optimizer.zero_grad()
            output = model(coords)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        #Avaliação
        model.eval()
        epoch_test_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                coords, target = batch['coordinates'], batch['distance']
                coords = coords.view(len(target), -1, 3)
                output = model(coords)
                
                all_preds.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
                epoch_test_loss += criterion(output, target).item()
        
        # Cálculo de métricas 
        epoch_train_loss /= len(train_loader)
        epoch_test_loss /= len(test_loader)
        
        all_targets = np.array(all_targets) * normalize_params['distance']['std'] + normalize_params['distance']['mean']
        all_preds = np.array(all_preds) * normalize_params['distance']['std'] + normalize_params['distance']['mean']
        
        r2 = r2_score(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        
        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)
        r2_scores.append(r2)
        mae_scores.append(mae)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}")
        print(f"  Test Loss: {epoch_test_loss:.4f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        if (epoch+1) % 10 == 0 or epoch == epochs-1:
            plot_metrics(train_losses, test_losses, r2_scores, mae_scores, all_preds, all_targets, plot_dir=model_dir)
            save_checkpoint(epoch, model, optimizer, train_losses, test_losses, r2_scores, mae_scores, checkpoint_dir)

    plot_metrics(train_losses, test_losses, r2_scores, mae_scores, all_preds, all_targets, plot_dir=model_dir)
    
    return model, test_dataset, (train_losses, test_losses, r2_scores, mae_scores)

def plot_metrics(train_losses, test_losses, r2_scores, mae_scores, all_preds, all_targets, plot_dir = None):
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

    if plot_dir:
        file_name = plot_dir + f"/training_metrics.png"
    else:
        file_name = f"training_metrics.png"
    
    plt.savefig(file_name)
    plt.close()
    print(f"Gráficos de métricas salvos em {file_name}")

def evaluate_model(model, test_dataset, plot_dir = None):
    print("\nAvaliando modelo no conjunto de teste...")
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for sample in test_dataset:
            coords = sample['coordinates'].unsqueeze(0)  # Adiciona dimensão de batch
            pred = model(coords, denormalize=True).item()
            
            true = sample['distance'].item() * test_dataset.normalize_params['distance']['std'] + \
                   test_dataset.normalize_params['distance']['mean']
            
            all_preds.append(pred)
            all_targets.append(true)
    
    # Restante da função permanece igual...
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

    if plot_dir:
        file_name = plot_dir + f"/predictions_vs_actuals.png"
    else:
        file_name = f"predictions_vs_actuals.png"

    plt.savefig(file_name)
    plt.close()
    print(f"Gráfico de predições vs reais salvo em {file_name}")

def predict_distance(model, pointcloud, num_points=1000):
    print("\nFazendo predição para nova pointcloud...")
    if len(pointcloud) > num_points:
        pointcloud = pointcloud[np.random.choice(len(pointcloud), num_points, replace=False)]
    
    coords = pointcloud[:, :3]
    coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-8)
    coords = torch.FloatTensor(coords).unsqueeze(0)  # Adiciona dimensão de batch
    
    with torch.no_grad():
        model.eval()
        pred = model(coords, denormalize=True)
    
    print(f"Predição concluída: {pred.item():.2f}")
    return pred.item()


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "training_data", "complete_training_data.npz")
    model_dir = os.path.join(os.path.dirname(__file__), "..", "data", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Treina o modelo
        # Aumentar o número de épocas e pontos para treino, e aumentar o tamanho da rede
        model, test_dataset, metrics = train_model(
            data_path,
            epochs=NUM_EPOCHS,
            batch_size=64,
            num_points=NUM_POINTS,
            model_dir=model_dir
        )
        
        # Salva o modelo
        torch.save(model.state_dict(), os.path.join(model_dir, "lane_distance_regressor.pth"))
        print("Modelo treinado e salvo com sucesso.")

        checkpoint_dir = os.path.join(model_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

        # Avaliação
        evaluate_model(model, test_dataset, plot_dir=model_dir)

        # Exemplo de predição
        sample_data = np.load(data_path, allow_pickle=True)
        sample_pc = sample_data['train'].item()['pointclouds'][0]
        distance = predict_distance(model, sample_pc, num_points=NUM_POINTS)
        print(f"\nExemplo de Predição: {distance:.2f} m")
        
    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")
        raise