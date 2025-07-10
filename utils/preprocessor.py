import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
from scipy.spatial import KDTree
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
from scipy.interpolate import splev
from collections import defaultdict

# Constantes
VELODYNE_TAG = 'VELODYNE_PARTIAL_SCAN_IN_FILE'
TIME_WINDOW = 0.2  # Janela temporal para agrupamento (em segundos)
SAMPLES_PER_FILE = 200

def load_all_globalpos_old(data_dir):
    """Carrega todas as posições globais dos arquivos .npy"""
    print("\n[1/4] Carregando dados de posição global...")
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
    all_data = []
    
    for filename in all_files:
        filepath = os.path.join(data_dir, filename)
        print(f"Extraindo globalpos do arquivo: {filename}")
        try:
            with open(filepath, 'rb') as f:
                dataset = np.load(f, allow_pickle=True)
                if len(dataset) == 0:
                    print(f"Arquivo vazio: {filename}")
                    continue
                
                # Calcula timestamp relativo
                timestamps = np.array([item['timestamp'] for item in dataset])
                relative_timestamps = timestamps - timestamps[0]
                
                for i, item in enumerate(dataset):
                    all_data.append({
                        'x': item['x'],
                        'y': item['y'],
                        'theta': item['theta'],
                        'global_timestamp': item['timestamp'],
                        'relative_timestamp': relative_timestamps[i],
                        'filename': filename,
                        'file_index': i
                    })
                    
        except Exception as e:
            print(f"Erro ao carregar {filename}: {str(e)}")
    
    print(f"Total de pontos carregados: {len(all_data)}")
    return all_data

def calculate_median_path_old(all_globalpos):
    """Calcula caminho mediano considerando tempo e orientação"""
    print("\n[2/4] Calculando caminho mediano...")
    
    # Agrupa por tempo e orientação
    time_groups = defaultdict(list)
    for item in all_globalpos:
        time_key = round(item['relative_timestamp'] / TIME_WINDOW)
        theta_key = round(np.degrees(item['theta']) / 10)  # Agrupa a cada 10 graus
        time_groups[(time_key, theta_key)].append(item)
    
    # Calcula medianas
    median_points = []
    for group_key in sorted(time_groups.keys()):
        group = time_groups[group_key]
        median_points.append({
            'x': np.median([p['x'] for p in group]),
            'y': np.median([p['y'] for p in group]),
            'theta': np.median([p['theta'] for p in group]),
            'relative_timestamp': np.median([p['relative_timestamp'] for p in group]),
            'n_points': len(group)
        })
    
    # Suavização
    if len(median_points) > 5:
        median_points = smooth_path_old(median_points)
    
    print(f"Caminho mediano calculado com {len(median_points)} pontos")
    return median_points

def smooth_path_old(path, window_size=5, polyorder=2):
    """Aplica suavização Savitzky-Golay ao caminho"""
    if len(path) < window_size:
        return path
    
    x = np.array([p['x'] for p in path])
    y = np.array([p['y'] for p in path])
    theta = np.array([p['theta'] for p in path])
    
    x_smooth = savgol_filter(x, window_size, polyorder)
    y_smooth = savgol_filter(y, window_size, polyorder)
    theta_smooth = savgol_filter(theta, window_size, polyorder)
    
    for i, p in enumerate(path):
        p['x'] = x_smooth[i]
        p['y'] = y_smooth[i]
        p['theta'] = theta_smooth[i]
    
    return path

def sample_pointclouds(data_dir, split_line=None, test_side='left', test_ratio=0.2, samples_per_file=100):
    target_train = int(samples_per_file * (1 - test_ratio))
    target_test = samples_per_file - target_train

    sampled_train = []
    sampled_test = []
    
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('.npy'):
            continue

        print(f"Amostrando nuvens de pontos do arquivo: {filename}")
        data = np.load(os.path.join(data_dir, filename), allow_pickle=True)
            
        # Separação treino/teste baseada na posição geográfica
        train_points = []
        test_points = []
        
        for item in data:
            x, y = item['x'], item['y']
            m, b = split_line
            
            if m == 0:  # Linha vertical
                is_test_side = (x < b) if test_side == 'left' else (x > b)
            else:
                pos = y - (m*x + b)
                is_test_side = (pos > 0) if test_side in ['above','right'] else (pos < 0)
                
            if is_test_side:
                test_points.append(item)
            else:
                train_points.append(item)

        if len(train_points) > 0:
            sampled_train.extend(np.random.choice(train_points, size=min(target_train, len(train_points)), replace=False))  

        if len(test_points) > 0:
            sampled_test.extend(np.random.choice(test_points, size=min(target_test, len(test_points)), replace=False))
        
    print(f"\nDivisão final:")
    print(f"Total de amostras: {len(sampled_train + sampled_test)}")
    print(f"Treino: {len(sampled_train)}")
    print(f"Teste: {len(sampled_test)}")
    
    return sampled_train, sampled_test

def calculate_distance_to_path(point, theta_vehicle, median_path):
    centroids = median_path['centroids']
    lines_params = median_path['lines_params']
    
    # 1. Encontra o centróide mais próximo
    distances_to_centroids = np.linalg.norm(centroids - point, axis=1)
    
    nearest_idx = np.argmin(distances_to_centroids)
    
    line = lines_params[nearest_idx]
    
    # 2. Calcula ponto mais próximo no segmento
    if line['type'] == 'line':
        if line['representation'] == 'y=f(x)':
            slope, intercept = line['params']
            # Projeção ortogonal para y = mx + b
            x_proj = (point[0] + slope*(point[1] - intercept))/(1 + slope**2)
            y_proj = slope*x_proj + intercept
            closest_point = np.array([x_proj, y_proj])
            seg_vec = np.array([1, slope])  # Vetor diretor da reta
        else:
            slope, intercept = line['params']
            # Projeção ortogonal para x = my + b
            y_proj = (point[1] + slope*(point[0] - intercept))/(1 + slope**2)
            x_proj = slope*y_proj + intercept
            closest_point = np.array([x_proj, y_proj])
            seg_vec = np.array([slope, 1])  # Vetor diretor da reta
    else:
        # Caso degenerado (apenas 1 ponto)
        closest_point = np.array([line['point'][0], line['point'][1]])
        seg_vec = np.array([1, 0])  # Vetor arbitrário
    
    # 3. Calcula distância com sinal
    point_vec = point - closest_point
    distance = np.linalg.norm(point_vec)
    cross = seg_vec[0]*point_vec[1] - seg_vec[1]*point_vec[0]
    signed_distance = np.sign(cross) * distance
    
    # 4. Ponderação pela orientação do veículo
    seg_angle = np.arctan2(seg_vec[1], seg_vec[0])
    angle_diff = min(abs(theta_vehicle - seg_angle), 
                   abs(theta_vehicle - seg_angle - np.pi))
    angular_weight = np.cos(angle_diff)
    
    return signed_distance * angular_weight, nearest_idx

def calculate_distances(sampled_data, median_path):
    theta_vehicle = 0.0
    for item in sampled_data:
        point = np.array([item['x'], item['y']])
        
        distance, segment = calculate_distance_to_path(point, theta_vehicle, median_path)
        
        item['distance'] = distance
        item['path_segment'] = segment
    
    return sampled_data

def plot_with_distances(npz_data, median_path, split_line=None, test_side='left', 
                       plot_dir=None, max_points_to_plot=50):
    plt.figure(figsize=(15, 10))
    line_color = 'black'
    
    # Extrai os dados do NPZ
    train_data = npz_data['train'].item() if isinstance(npz_data['train'], np.ndarray) else npz_data['train']
    test_data = npz_data['test'].item() if isinstance(npz_data['test'], np.ndarray) else npz_data['test']
    
    train_positions = train_data['positions']
    train_distances = train_data['distances']
    train_segments = train_data['metadata'] if 'metadata' in train_data else np.zeros_like(train_distances)
    
    test_positions = test_data['positions']
    test_distances = test_data['distances']
    test_segments = test_data['metadata'] if 'metadata' in test_data else np.zeros_like(test_distances)
    
    # Combina todos os pontos para normalização
    all_points = np.vstack([train_positions[:, :2], test_positions[:, :2]])
    centroids = median_path['centroids']
    lines_params = median_path['lines_params']
    
    # 1. Plot das linhas do caminho médio
    for i, line in enumerate(lines_params):
        if line['type'] == 'spline':
            u_vals = np.linspace(line['u_range'][0], line['u_range'][1], 20)
            x, y = splev(u_vals, line['tck'])
            plt.plot(x, y, color=line_color, linewidth=1.5)
        elif line['type'] == 'line':
            if line['representation'] == 'y=f(x)':
                plt.plot(line['x_range'], line['y_range'], 
                        color=line_color, linewidth=1.5)
            else:
                plt.plot(line['x_range'], line['y_range'],
                        color=line_color, linewidth=1.5)
        elif line['type'] == 'point':
            plt.scatter(line['point'][0], line['point'][1], 
                       color=line_color, s=50, marker='s')
    
    # 2. Plot dos centróides
    for i, centroid in enumerate(centroids):
        plt.scatter(centroid[0], centroid[1], color='blue', marker='x', s=25)
        plt.text(centroid[0], centroid[1], f'{i}', fontsize=10, 
                ha='center', va='bottom', color='black')
    
    # 3. Adiciona a reta divisória
    if split_line is not None:
        m, b = split_line
        if m == 0:  # Linha vertical
            x_val = b
            y_min = np.min(all_points[:, 1])
            y_max = np.max(all_points[:, 1])
            plt.plot([x_val, x_val], [y_min, y_max], 
                    'r--', linewidth=3, label=f'Divisão Treino/Teste (x = {b})')
        else:
            x_vals = np.array([np.min(all_points[:, 0]), np.max(all_points[:, 0])])
            y_vals = m * x_vals + b
            plt.plot(x_vals, y_vals, 'r--', linewidth=3, label='Divisão Treino/Teste')
    
    # 4. Plot dos pontos (treino e teste)
    def plot_points_group(positions, distances, segments, color, label, max_points):
        for i, (pos, dist, seg) in enumerate(zip(positions[:max_points], 
                                               distances[:max_points], 
                                               segments[:max_points])):
            centroid = centroids[seg]
            
            # Plot do ponto
            plt.scatter(pos[0], pos[1], c=color,
                       s=80, alpha=0.7, edgecolors='black',
                       label=label if i == 0 else "")
            
            # Linha para o centróide
            plt.plot([pos[0], centroid[0]], [pos[1], centroid[1]],
                    'k-', linewidth=1, alpha=0.5)
            
            # Anotação
            plt.annotate(f'{label[0]}{i}\n{dist:.1f}m',
                        xy=(pos[0], pos[1]),
                        xytext=(5,5), textcoords='offset points',
                        fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    
    # Plot pontos de treino e teste
    plot_points_group(train_positions, train_distances, train_segments, 
                     'green', 'Treino', max_points_to_plot//2)
    plot_points_group(test_positions, test_distances, test_segments,
                     'red', 'Teste', max_points_to_plot//2)
    
    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.grid(True, alpha=0.3)
    plt.title('Distâncias ao Caminho Médio')
    plt.legend()
    
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{plot_dir}/distancias_centroides_{timestamp}.png"
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        print(f"Gráfico salvo em: {file_name}")
        plt.close()
    else:
        plt.show()

def save_training_data(train_data, test_data, output_dir):

    if not output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Organização dos dados em subconjuntos
    dataset_dict = {
        'train': {
            'pointclouds': [],
            'distances': [],
            'positions': [],
            'metadata': []
        },
        'test': {
            'pointclouds': [],
            'distances': [],
            'positions': [],
            'metadata': []
        }
    }
    
    # Processamento do conjunto de treino
    for sample in train_data:
        dataset_dict['train']['pointclouds'].append(sample['pointcloud'].astype(np.float32))
        dataset_dict['train']['distances'].append(np.float32(sample['distance']))
        dataset_dict['train']['positions'].append([
            np.float32(sample['x']),
            np.float32(sample['y']),
            np.float32(sample['theta'])
        ])
        dataset_dict['train']['metadata'].append(np.int32(sample.get('path_segment', 0)))
    
    # Processamento do conjunto de teste
    for sample in test_data:
        dataset_dict['test']['pointclouds'].append(sample['pointcloud'].astype(np.float32))
        dataset_dict['test']['distances'].append(np.float32(sample['distance']))
        dataset_dict['test']['positions'].append([
            np.float32(sample['x']),
            np.float32(sample['y']),
            np.float32(sample['theta'])
        ])
        dataset_dict['test']['metadata'].append(np.int32(sample.get('path_segment', 0)))
    
    # Conversão para arrays numpy otimizados
    compiled_data = {
        'train': {
            'pointclouds': np.array(dataset_dict['train']['pointclouds'], dtype=object),
            'distances': np.array(dataset_dict['train']['distances'], dtype=np.float32),
            'positions': np.array(dataset_dict['train']['positions'], dtype=np.float32),
            'metadata': np.array(dataset_dict['train']['metadata'], dtype=np.int32)
        },
        'test': {
            'pointclouds': np.array(dataset_dict['test']['pointclouds'], dtype=object),
            'distances': np.array(dataset_dict['test']['distances'], dtype=np.float32),
            'positions': np.array(dataset_dict['test']['positions'], dtype=np.float32),
            'metadata': np.array(dataset_dict['test']['metadata'], dtype=np.int32)
        },
        # Metadados globais
        'metadata': np.array([{
            'creation_date': datetime.datetime.now().isoformat(),
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'train_mean_distance': float(np.mean(dataset_dict['train']['distances'])),
            'test_mean_distance': float(np.mean(dataset_dict['test']['distances'])),
            'version': '1.2',
            'split_info': {
                'method': 'geographic',
                'split_line': train_data[0].get('split_line', None) if train_data else None,
                'test_side': train_data[0].get('test_side', None) if train_data else None
            }
        }], dtype=object)
    }
    
    # Salvamento do arquivo
    output_path = os.path.join(output_dir, "complete_training_data.npz")
    np.savez_compressed(output_path, 
                    train=compiled_data['train'],
                    test=compiled_data['test'],
                    metadata=compiled_data['metadata'])
    
    # Relatório detalhado
    print("\n" + "="*60)
    print("Dataset salvo com sucesso")
    print("="*60)
    print(f"Arquivo: {output_path}")

def view_training_data(npz_file_path, num_samples=5):
    try:
        data = np.load(npz_file_path, allow_pickle=True)
        
        print("\n" + "="*60)
        print("VISUALIZAÇÃO DE DADOS DE TREINO/TESTE")
        print("="*60)
        print(f"Arquivo: {npz_file_path}")
        
        # Detecta a estrutura do arquivo
        if 'train' in data and 'test' in data:
            # Versão com dicionários aninhados
            train_data = data['train'].item()
            test_data = data['test'].item()
            metadata = data['metadata'].item() if 'metadata' in data else {}
            structure = "nested"
        else:
            # Versão com chaves prefixadas
            train_data = {k[6:]: data[k] for k in data if k.startswith('train_')}
            test_data = {k[5:]: data[k] for k in data if k.startswith('test_')}
            metadata = data['metadata'].item() if 'metadata' in data else {}
            structure = "prefixed"
        
        # 1. Mostra metadados
        print("\n[1/4] METADADOS:")
        print(f"Estrutura detectada: {structure}")
        print(f"Data de criação: {metadata.get('creation_date', 'N/A')}")
        print(f"Versão: {metadata.get('version', 'N/A')}")
        print(f"Amostras treino: {metadata.get('train_samples', len(train_data.get('pointclouds', [])))}")
        print(f"Amostras teste: {metadata.get('test_samples', len(test_data.get('pointclouds', [])))}")

        # 2. Estatísticas básicas
        print("\n[2/4] ESTATÍSTICAS:")
        
        # Treino
        if 'pointclouds' in train_data:
            print("\nCONJUNTO DE TREINO:")
            print(f"Nuvens de pontos: {len(train_data['pointclouds'])}")
            print(f"Distância média: {np.mean(train_data['distances']):.3f}m")
            print(f"Pontos/nuvem (média): {np.mean([pc.shape[0] for pc in train_data['pointclouds']]):.1f}")
        
        # Teste
        if 'pointclouds' in test_data:
            print("\nCONJUNTO DE TESTE:")
            print(f"Nuvens de pontos: {len(test_data['pointclouds'])}")
            print(f"Distância média: {np.mean(test_data['distances']):.3f}m")
            print(f"Pontos/nuvem (média): {np.mean([pc.shape[0] for pc in test_data['pointclouds']]):.1f}")

        # 3. Amostras de dados
        print(f"\n[3/4] AMOSTRAS (mostrando {num_samples} de cada):")
        
        def print_sample(sample, prefix, idx):
            print(f"\n{prefix} {idx+1}:")
            if 'positions' in sample:
                print(f"Posição: (x={sample['positions'][idx][0]:.2f}, y={sample['positions'][idx][1]:.2f}, θ={sample['positions'][idx][2]:.3f})")
            print(f"Distância: {sample['distances'][idx]:.3f}m")
            print(f"Segmento: {sample.get('metadata', [0]*len(sample['distances']))[idx]}")
            print("Pointcloud (3 primeiros pontos):")
            print(sample['pointclouds'][idx][:3])

        # Amostras de treino
        if 'pointclouds' in train_data and len(train_data['pointclouds']) > 0:
            indices = np.random.choice(len(train_data['pointclouds']), 
                                     size=min(num_samples, len(train_data['pointclouds'])), 
                                     replace=False)
            for i, idx in enumerate(indices):
                print_sample(train_data, "Treino", i)

        # Amostras de teste
        if 'pointclouds' in test_data and len(test_data['pointclouds']) > 0:
            indices = np.random.choice(len(test_data['pointclouds']), 
                                     size=min(num_samples, len(test_data['pointclouds'])), 
                                     replace=False)
            for i, idx in enumerate(indices):
                print_sample(test_data, "Teste", i)

        # 4. Verificação de integridade
        print("\n[4/4] VERIFICAÇÃO:")
        checks_passed = True
        
        if 'pointclouds' in train_data and len(train_data['pointclouds']) != len(train_data['distances']):
            print("Inconsistência no treino: Número de pointclouds != distâncias")
            checks_passed = False
            
        if 'pointclouds' in test_data and len(test_data['pointclouds']) != len(test_data['distances']):
            print("Inconsistência no teste: Número de pointclouds != distâncias")
            checks_passed = False
            
        if checks_passed:
            print("Todos os checks de integridade passaram")

    except Exception as e:
        print(f"\nErro durante a visualização: {str(e)}")
        raise
    finally:
        if 'data' in locals():
            data.close()


def main():
    print("="*60)
    print(" PRÉ-PROCESSAMENTO DE DADOS LIDAR ")
    print("="*60)

    # Configura caminhos
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "output")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "training_data")

    # Pipeline principal
    #globalpos_data = load_all_globalpos_old(data_dir)
    #median_path = calculate_median_path_old(globalpos_data)
   
    median_data = np.load(os.path.join(data_dir, "caminho_mediano", 'caminho_mediano.npz'), allow_pickle=True)

    median_path = {
        'x_offset': median_data['x_offset'],
        'y_offset': median_data['y_offset'],
        'centroids': median_data['centroids'],
        'lines_params': median_data['lines_params'],
        'split_params': median_data['split_params'][()] if 'split_params' in median_data else None
    }

    print(f"\nArquivo {os.path.join(data_dir, "caminho_mediano") + '/caminho_mediano.npz'} contendo o Caminho Mediano lido.")

    split_params = median_path['split_params']
    
    train_samples, test_samples = sample_pointclouds(data_dir, split_line=split_params['split_line'], 
                                test_side=split_params['test_side'], 
                                test_ratio=0.2, 
                                samples_per_file=SAMPLES_PER_FILE)
    # Calcula distâncias (agora opera em todos os dados)
    print("\nCalculando distâncias usando projeção ortogonal nas retas dos centróides...")
    train_samples = calculate_distances(train_samples, median_path)
    test_samples = calculate_distances(test_samples, median_path)

    complete_data_path = os.path.join(output_dir, "complete_training_data.npz")


    # Salva os dados
    save_training_data(train_samples, test_samples, output_dir)
    view_training_data(complete_data_path)

    # Carrega os dados completos de treinamento
    complete_data = np.load(complete_data_path, allow_pickle=True)

    plot_with_distances(complete_data, median_path, 
                   split_line=median_path['split_params']['split_line'], 
                   test_side=median_path['split_params']['test_side'],
                   plot_dir=output_dir)
    
    

if __name__ == "__main__":
    main()