import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy import inf
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torchviz import make_dot

NUM_EPOCHS = 3000
NUM_POINTS = 8000
PATIENCE = 0 #Critério para parada prévia, mantenha em 0 para desativar
MIN_DELTA = 0.001
MONITOR_METRIC = 'test_loss' # Pode mudar para 'r2_score', 'test_mae', etc.
OPTIMIZER_LR = 0.01 #Learning rate inicial do otimizador

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.001, mode='min', monitor_metric='test_loss', checkpoint_dir=None):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor_metric = monitor_metric
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_metrics = {}
        
        # Mapeamento de métricas e seus modos ideais
        self.metric_modes = {
            'test_loss': 'min',
            'test_mae': 'min', 
            'test_rmse': 'min',
            'r2_score': 'max'
        }
        
        if self.monitor_metric not in self.metric_modes:
            raise ValueError(f"Métrica {monitor_metric} não suportada. Use: {list(self.metric_modes.keys())}")
        
        self._load_best_score_from_checkpoint()
    
    def __call__(self, current_metrics, model, optimizer, scheduler, train_losses, test_losses, r2_scores, mae_scores, rmse_scores):
        if self.monitor_metric not in current_metrics:
            raise ValueError(f"Métrica {self.monitor_metric} não encontrada nas métricas fornecidas")
                
        current_score = current_metrics[self.monitor_metric]
        expected_mode = self.metric_modes[self.monitor_metric]
        
        if self.best_score is None:
            self.best_score = current_score
            self._save_checkpoint(current_metrics, model, optimizer, scheduler, train_losses, test_losses, r2_scores, mae_scores, rmse_scores)
            print(f"\nMelhor pontuação de {self.monitor_metric} inicial: {self.best_score:.6f}")
            self.early_stop = False
        else:
            # Verifica se há melhoria
            if expected_mode == 'min':
                if current_score < self.best_score - self.min_delta:
                    print(f"\nMelhoria detectada, pontuação de {self.monitor_metric} anterior: {self.best_score:.6f} -> pontuação atual: {current_score:.6f}")
                    self.best_score = current_score
                    self.counter = 0
                    self._save_checkpoint(current_metrics, model, optimizer, scheduler, train_losses, test_losses, r2_scores, mae_scores, rmse_scores)
            elif expected_mode == 'max':
                if current_score > self.best_score + self.min_delta:
                    print(f"\nMelhoria detectada, pontuação de {self.monitor_metric} anterior: {self.best_score:.6f} -> pontuação atual: {current_score:.6f}")
                    self.best_score = current_score
                    self.counter = 0
                    self._save_checkpoint(current_metrics, model, optimizer, scheduler, train_losses, test_losses, r2_scores, mae_scores, rmse_scores)
            else:    
                self.counter += 1
                if self.counter == self.patience:
                    self.early_stop = True

        return self.early_stop

        
    
    def _load_best_score_from_checkpoint(self):
        #Carrega o melhor score de um checkpoint existente ao retomar treinamento
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint_pointnet.pth")
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
                print(f"\nCheckpoint de melhor modelo encontrado em {checkpoint_path}")
                
                # Encontra o melhor valor histórico da métrica monitorada
                if self.monitor_metric == 'r2_score' and 'r2_scores' in checkpoint and checkpoint['r2_scores']:
                    best_value = max(checkpoint['r2_scores'])
                    print(f"  - Melhor R² Score histórico: {best_value:.6f}")
                elif self.monitor_metric == 'test_loss' and 'test_losses' in checkpoint and checkpoint['test_losses']:
                    best_value = min(checkpoint['test_losses'])
                    print(f"  - Melhor Test Loss histórico: {best_value:.6f}")
                elif self.monitor_metric == 'test_mae' and 'mae_scores' in checkpoint and checkpoint['mae_scores']:
                    best_value = min(checkpoint['mae_scores'])
                    print(f"  - Melhor MAE histórico: {best_value:.6f}")
                elif self.monitor_metric == 'test_rmse' and 'rmse_scores' in checkpoint and checkpoint['rmse_scores']:
                    best_value = min(checkpoint['rmse_scores'])
                    print(f"  - Melhor RMSE histórico: {best_value:.6f}")
                else:
                    # Fallback: usa o valor mais recente se disponível
                    if self.monitor_metric in checkpoint:
                        best_value = checkpoint[self.monitor_metric]
                        print(f"  - Valor mais recente da métrica: {best_value:.6f}")
                    else:
                        best_value = None
                        print("  - Métrica não encontrada no checkpoint")
                
                if best_value is not None:
                    self.best_score = best_value
                    # Ajusta para maximização se necessário
                    expected_mode = self.metric_modes[self.monitor_metric]
                    print(f"  - Best_score configurado para: {best_value:.6f} (modo: {expected_mode})")
                    
                    # Carrega também as melhores métricas
                    if 'epoch' in checkpoint:
                        self.best_metrics = {
                            'epoch': checkpoint['epoch'],
                            'train_loss': checkpoint['train_losses'][-1] if checkpoint['train_losses'] else 0,
                            'test_loss': checkpoint['test_losses'][-1] if checkpoint['test_losses'] else 0,
                            'r2_score': checkpoint['r2_scores'][-1] if checkpoint['r2_scores'] else 0,
                            'test_mae': checkpoint['mae_scores'][-1] if checkpoint['mae_scores'] else 0,
                            'test_rmse': checkpoint['rmse_scores'][-1] if checkpoint['rmse_scores'] else 0,
                        }
                        print(f"  - Época do melhor modelo: {checkpoint['epoch']}")
                else:
                    print("  - Nenhum valor válido encontrado no checkpoint")
                        
            except Exception as e:
                print(f"Erro ao carregar melhor score do checkpoint: {e}")
        else:
            print(f"Nenhum checkpoint de melhor modelo encontrado em {checkpoint_path}. Iniciando do zero.")

    
    def _save_checkpoint(self, current_metrics, model, optimizer, scheduler, train_losses, test_losses, r2_scores, mae_scores, rmse_scores):
            save_checkpoint(
            epoch=current_metrics['epoch'],
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_losses=train_losses,
            test_losses=test_losses,
            r2_scores=r2_scores,
            mae_scores=mae_scores,
            rmse_scores=rmse_scores,
            checkpoint_dir=self.checkpoint_dir,
        )
            self.best_metrics = current_metrics.copy()
    
    def load_best_model(self, model, optimizer=None, scheduler=None):
        if self.checkpoint_dir:
            checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint_pointnet.pth")
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                
                if optimizer and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if scheduler and checkpoint['scheduler_state_dict']:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                return True
        return False

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
    

def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, test_losses, r2_scores, mae_scores, rmse_scores, checkpoint_dir):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'r2_scores': r2_scores,
        'mae_scores': mae_scores,
        'rmse_scores': rmse_scores,
    }
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_pointnet.pth')
    torch.save(checkpoint, checkpoint_path)

def load_latest_checkpoint(checkpoint_dir, model, optimizer, scheduler):
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_pointnet.pth')
    
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return (
        checkpoint['epoch'],
        checkpoint['train_losses'],
        checkpoint['test_losses'],
        checkpoint['r2_scores'],
        checkpoint['mae_scores'],
        checkpoint['rmse_scores']
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
    best_model = PointNetPP(
        norm_mean=normalize_params['distance']['mean'],
        norm_std=normalize_params['distance']['std']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=OPTIMIZER_LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()

    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    best_checkpoint_dir = os.path.join(checkpoint_dir, "best_checkpoint")
    train_losses, test_losses, r2_scores, mae_scores, rmse_scores = [], [], [], [], []

    start_epoch = 0

    early_stopping = None
    early_stopping = EarlyStopping(
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        monitor_metric=MONITOR_METRIC,  
        mode='min',
        checkpoint_dir=best_checkpoint_dir,
    )

    if os.path.exists(os.path.join(checkpoint_dir, 'checkpoint_pointnet.pth')):
        start_epoch, train_losses, test_losses, r2_scores, mae_scores, rmse_scores = load_latest_checkpoint(checkpoint_dir, model, optimizer, scheduler)
        print(f"Checkpoint carregado: {checkpoint_dir} (época {start_epoch + 1})\nRetomando o treinamento...")

    print("Iniciando treinamento...")
    for epoch in range(start_epoch, epochs):
        # Treinamento
        model.train()
        epoch_train_loss = 0

        print('\n')
        train_pbar = tqdm(train_loader, desc=f"Época {epoch+1}/{epochs} [Treino]", unit="batch", ascii=True)

        for batch in train_pbar:
            coords, target = batch['coordinates'], batch['distance']
            coords = coords.view(len(target), -1, 3)
            
            optimizer.zero_grad()
            output = model(coords)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()

            train_pbar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'epoch_loss': f'{epoch_train_loss/len(train_loader):.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

        train_pbar.close()

        # Avaliação
        model.eval()
        epoch_test_loss = 0
        all_preds = []
        all_targets = []

        print('\n')
        test_pbar = tqdm(test_loader, desc=f"Época {epoch+1}/{epochs} [Teste]", unit="batch", ascii=True)

        with torch.no_grad():
            for batch in test_pbar:
                coords, target = batch['coordinates'], batch['distance']
                coords = coords.view(len(target), -1, 3)
                output = model(coords)
                
                all_preds.extend(output.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
                batch_loss = criterion(output, target).item()
                epoch_test_loss += batch_loss

                test_pbar.set_postfix({
                    'batch_loss': f'{batch_loss:.4f}',
                    'epoch_loss': f'{epoch_test_loss/len(test_loader):.4f}'
                })

        test_pbar.close()
        
        # Cálculo de métricas 
        epoch_train_loss /= len(train_loader)
        epoch_test_loss /= len(test_loader)
        
        all_targets = np.array(all_targets) * normalize_params['distance']['std'] + normalize_params['distance']['mean']
        all_preds = np.array(all_preds) * normalize_params['distance']['std'] + normalize_params['distance']['mean']
        
        r2 = r2_score(all_targets, all_preds)
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        
        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)

        scheduler.step(epoch_test_loss)

        current_metrics = {
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'test_loss': epoch_test_loss,
            'test_mae': mae,
            'test_rmse': rmse,
            'r2_score': r2,
            'learning_rate': optimizer.param_groups[0]['lr']
        }

        if early_stopping is not None:
            if early_stopping(current_metrics, model, optimizer, scheduler, train_losses, test_losses, r2_scores, mae_scores, rmse_scores):
                print("Early stopping acionado!")
                break
            
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}")
        print(f"  Test Loss: {epoch_test_loss:.4f}")
        print(f"  R² Score: {r2:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        if (epoch+1) % 10 == 0 or epoch == epochs-1:
            plot_metrics(train_losses, test_losses, r2_scores, mae_scores, all_preds, all_targets, plot_dir=model_dir)
            save_checkpoint(epoch, model, optimizer, scheduler, train_losses, test_losses, r2_scores, mae_scores, rmse_scores, checkpoint_dir)
            print(f"Checkpoint salvo em {checkpoint_dir}")

    plot_metrics(train_losses, test_losses, r2_scores, mae_scores, all_preds, all_targets, plot_dir=model_dir)

    early_stopping.load_best_model(best_model, optimizer)
    
    return best_model, model, test_dataset, (train_losses, test_losses, r2_scores, mae_scores, rmse_scores)

def plot_metrics(train_losses, test_losses, r2_scores, mae_scores, all_preds, all_targets, plot_dir = None):
    plt.figure(figsize=(20, 10))
    
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
    print(f"\nGráficos de métricas salvos em {file_name}")

def evaluate_model(model, test_dataset, plot_dir = None, timestamp = None):
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
    plt.figure(figsize=(15, 10))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    plt.plot([min(all_targets), max(all_targets)], [min(all_targets), max(all_targets)], 'r--')
    plt.xlabel('Valores Reais')
    plt.ylabel('Predições')
    plt.title('Predições vs Valores Reais')
    plt.grid(True)

    if plot_dir:
        file_name = plot_dir + f"/predictions_vs_actuals_{timestamp}.png"
    else:
        file_name = f"predictions_vs_actuals_{timestamp}.png"

    plt.savefig(file_name)
    plt.close()
    print(f"Gráfico de predições vs reais salvo em {file_name}")

     # Pega uma amostra do dataset de teste para visualizar o forward pass
    sample = test_dataset[0]
    coords = sample['coordinates'].unsqueeze(0)
    
    # Forward pass para capturar o grafo computacional
    output = model(coords, denormalize=True)
    
    # Cria a visualização
    dot = make_dot(output, params=dict(model.named_parameters()))
    
    # Salva o gráfico
    if plot_dir:
        model_viz_file = plot_dir + f"/model_architecture_{timestamp}"
    else:
        model_viz_file = f"model_architecture_{timestamp}"
    
    # Salva em formato PNG e DOT
    dot.render(model_viz_file, format='png', cleanup=True)
    print(f"Visualização da arquitetura do modelo salva em {model_viz_file}.png")

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
        best_model, model, test_dataset, (train_losses, test_losses, r2_scores, mae_scores, rmse_scores) = train_model(
            data_path,
            epochs=NUM_EPOCHS,
            batch_size=64,
            num_points=NUM_POINTS,
            model_dir=model_dir
        )
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Salva o modelo
        torch.save(model.state_dict(), os.path.join(model_dir, f"latest_lane_distance_regressor_{timestamp}.pth"))
        torch.save(best_model.state_dict(), os.path.join(model_dir, f"best_lane_distance_regressor_{timestamp}.pth"))
        print(f"Modelos treinados e salvos com sucesso no diretório {model_dir}\n")

        #Armazenando métricas:
        metrics_file = os.path.join(model_dir, f"training_metrics_{timestamp}.npz")

        #Salva no arquivo NPZ
        np.savez(metrics_file,
                 train_losses=np.array(train_losses),
                 test_losses=np.array(test_losses),
                 r2_scores=np.array(r2_scores),
                 mae_scores=np.array(mae_scores),
                 rmse_scores=np.array(rmse_scores),
                 timestamp=timestamp,
                 num_epochs=len(train_losses),
                 final_epoch=len(train_losses))
        
        print(f"Métricas de treinamento salvas em: {metrics_file}")

        checkpoint_dir = os.path.join(model_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

        # Avaliação
        evaluate_model(best_model, test_dataset, plot_dir=model_dir, timestamp=timestamp)

        # Exemplo de predição
        sample_data = np.load(data_path, allow_pickle=True)
        sample_pc = sample_data['train'].item()['pointclouds'][0]
        distance = predict_distance(model, sample_pc, num_points=NUM_POINTS)
        print(f"\nExemplo de Predição: {distance:.2f} m")
        
    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")
        raise