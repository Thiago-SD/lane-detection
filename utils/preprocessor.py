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

def sample_pointclouds(data_dir, split_line=None, test_side='left', x_offset=0.0, y_offset=0.0):
    """Amostra pointclouds e já separa em treino/teste conforme linha de divisão"""
    train_data = []
    test_data = []
    
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith('.npy'):
            continue
            
        filepath = os.path.join(data_dir, filename)
        print(f"Amostrando pointclouds do arquivo: {filename}")
        try:
            with open(filepath, 'rb') as f:
                dataset = np.load(f, allow_pickle=True)
                if len(dataset) == 0:
                    continue
                
                # Amostra aleatória
                indices = np.random.choice(
                    len(dataset), 
                    size=min(SAMPLES_PER_FILE, len(dataset)), 
                    replace=False
                )
                
                for idx in indices:
                    item = dataset[idx]
                    rel_time = item['timestamp'] - dataset[0]['timestamp']
                    
                    sample = {
                        'pointcloud': item['pointcloud'],
                        'globalpos': np.array([item['x'], item['y'], item['theta']]),
                        'global_timestamp': item['timestamp'],
                        'relative_timestamp': rel_time,
                        'source_file': filename,
                        'original_index': idx
                    }
                    
                    # Separa em treino/teste conforme a linha
                    if split_line is not None:
                        m, b = split_line
                        x = item['x']
                        # Verifica posição relativa à linha
                        if (test_side == 'right' and x > b) or (test_side == 'left' and x < b):
                            test_data.append(sample)
                        else:
                            train_data.append(sample)
                    else:
                        train_data.append(sample)
                    
        except Exception as e:
            print(f"Erro ao processar {filename}: {str(e)}")
    
    print(f"Total de pointclouds amostradas - Treino: {len(train_data)}, Teste: {len(test_data)}")
    return train_data, test_data

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
    print("\nCalculando distâncias usando projeção ortogonal nas retas dos centróides...")
    theta_vehicle = 0.0
    for item in sampled_data:
        point = np.array([item['globalpos'][0], item['globalpos'][1]])
        theta_vehicle = item['globalpos'][2]
        
        distance, segment = calculate_distance_to_path(point, theta_vehicle, median_path)
        
        item['distance'] = distance
        item['path_segment'] = segment
    
    return sampled_data

def plot_with_distances(points_with_distances, median_path, split_line=None, test_side='left', 
                       plot_dir=None, max_points_to_plot=50):
    plt.figure(figsize=(15, 10))
    line_color = 'black'

    # Normalização relativa
    all_points = np.array([p['point'] for p in points_with_distances])
    centroids = median_path['centroids']
    lines_params = median_path['lines_params']
    offset = np.mean(np.vstack([all_points, centroids]), axis=0)
    scale = np.max(np.std(np.vstack([all_points, centroids]), axis=0))

    # Plot das linhas
    for i, line in enumerate(lines_params):
        if line['type'] == 'spline':
            # Spline paramétrica - funciona para qualquer orientação
            u_vals = np.linspace(line['u_range'][0], line['u_range'][1], 20)
            x, y = splev(u_vals, line['tck'])
            plt.plot(x, y, color=line_color, linewidth=1.5)
        elif line['type'] == 'line':
            # Linha reta - tratamento especial para verticais
            if line['representation'] == 'y=f(x)':
                plt.plot(line['x_range'], line['y_range'], 
                        color=line_color, linewidth=1.5)
            else:  # x=f(y)
                plt.plot(line['x_range'], line['y_range'],
                        color=line_color, linewidth=1.5)
                
        elif line['type'] == 'point':
            plt.scatter(line['point'][0], line['point'][1], 
                       color=line_color, s=50, marker='s')
    
    # 2. Plot dos centróides (normalizados)
    for i, centroid in enumerate(centroids):
        plt.scatter(centroid[0], centroid[1], color='blue', label=f'Centroide {i}', marker='x', s=25)
        plt.text(centroid[0], centroid[1], f'{i}', fontsize=10, ha='center', va='bottom', color='black')

    # 3. Adiciona a reta divisória se fornecida
    if split_line is not None:
        m, b = split_line
        
        # Para retas verticais
        if m == 0:
            x_val = b
            y_min = np.min(all_points[:, 1])
            y_max = np.max(all_points[:, 1])
            plt.plot([x_val, x_val], [y_min, y_max], 
                    'r--', linewidth=3, label=f'Divisão Treino/Teste (x = {b})')
        else:
            # Código existente para retas não verticais
            x_vals = np.array([np.min(all_points[:, 0]), np.max(all_points[:, 0])])
            y_vals = m * x_vals + b
            plt.plot(x_vals, y_vals, 'r--', linewidth=3, label='Divisão Treino/Teste')

    # 4. Plot dos pontos (normalizados)
    for i, point_data in enumerate(points_with_distances[:max_points_to_plot]):
        point = point_data['point']
        centroid = centroids[point_data['path_segment']]
        
        # Plot do ponto
        plt.scatter(point[0], point[1], 
                   c='green' if point_data['distance'] >=0 else 'red',
                   s=80, alpha=0.7, edgecolors='black')
        
        # Linha para o centróide
        plt.plot([point[0], centroid[0]], [point[1], centroid[1]],
                'k-', linewidth=1, alpha=0.5)
        
        # Anotação
        plt.annotate(f'P{i}\n{point_data["distance"]:.1f}m',
                    xy=point,
                    xytext=(5,5), textcoords='offset points',
                    fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    plt.gca().set_aspect('equal', adjustable='datalim')
    plt.grid(True, alpha=0.3)
    plt.title('Distâncias ao Caminho Médio (Coordenadas Normalizadas)')
    
    if plot_dir:
        plt.savefig(f"{plot_dir}/distancias_centroides_{datetime.datetime.now().timestamp()}.png", dpi=300, bbox_inches='tight')
        plt.close()

def save_training_data(train_data, test_data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Combina todos os dados
    all_data = train_data + test_data
    
    # Cria array de máscara (1 para treino, 0 para teste)
    train_mask = np.array([1] * len(train_data) + [0] * len(test_data), dtype=np.uint8)
    
    # Extrai todos os dados em arrays separados
    x = np.array([item['globalpos'][0] for item in all_data], dtype=np.float32)
    y = np.array([item['globalpos'][1] for item in all_data], dtype=np.float32)
    theta = np.array([item['globalpos'][2] for item in all_data], dtype=np.float32)
    global_timestamps = np.array([item['global_timestamp'] for item in all_data], dtype=np.float64)
    relative_timestamps = np.array([item['relative_timestamp'] for item in all_data], dtype=np.float64)
    distances = np.array([item['distance'] for item in all_data], dtype=np.float32)
    pointclouds = np.array([item['pointcloud'] for item in all_data], dtype=object)
    source_files = np.array([item['source_file'] for item in all_data], dtype=object)
    
    # Cria dicionário com metadados
    metadata = {
        'description': 'Dataset completo para detecção de faixa',
        'num_samples': len(all_data),
        'num_train': len(train_data),
        'num_test': len(test_data),
        'variables': ['x', 'y', 'theta', 'global_timestamp', 'relative_timestamp', 
                     'distance', 'pointcloud', 'source_file', 'train_mask'],
        'units': {
            'x': 'metros',
            'y': 'metros',
            'theta': 'radianos',
            'timestamps': 'segundos',
            'distance': 'metros'
        },
    }
    
    # Salva em um único arquivo .npz (comprimido)
    output_path = os.path.join(output_dir, "complete_training_data.npz")
    np.savez_compressed(
        output_path,
        x=x,
        y=y,
        theta=theta,
        global_timestamps=global_timestamps,
        relative_timestamps=relative_timestamps,
        distances=distances,
        pointclouds=pointclouds,
        source_files=source_files,
        train_mask=train_mask,
        metadata=metadata
    )
    
    print(f"\nDados salvos em: {output_path}")
    print(f"Total de amostras: {len(all_data)} (Treino: {len(train_data)}, Teste: {len(test_data)})")
    print("Variáveis incluídas:")
    print("- Posições (x, y, theta)")
    print("- Timestamps (global e relativo)")
    print("- Distâncias até o centro da faixa")
    print("- Pointclouds completas")
    print("- Arquivos de origem")
    print("- Máscara de treino/teste (train_mask)")

def view_training_data(npz_file_path, num_samples=5):
    """
    Visualiza a estrutura e conteúdo do arquivo de dados de treino .npz
    
    Args:
        npz_file_path: Caminho para o arquivo .npz
        num_samples: Número de amostras a exibir (padrão: 5)
    """
    try:
        # Carrega o arquivo .npz
        data = np.load(npz_file_path, allow_pickle=True)
        
        print("="*60)
        print("VISUALIZAÇÃO DE DADOS DE TREINO")
        print("="*60)
        
        # 1. Mostra estrutura geral
        print("\n[1/4] ESTRUTURA DO ARQUIVO:")
        print(f"Arquivo: {npz_file_path}")
        print("Variáveis disponíveis:")
        for key in data.keys():
            if key != 'metadata':
                print(f"- {key}: {data[key].shape} ({data[key].dtype})")
        
        # 2. Mostra metadados
        if 'metadata' in data:
            print("\n[2/4] METADADOS:")
            metadata = data['metadata'].item()
            for key, value in metadata.items():
                print(f"{key}: {value}")
        
        # 3. Mostra estatísticas básicas
        print("\n[3/4] ESTATÍSTICAS BÁSICAS:")
        if 'distances' in data:
            distances = data['distances']
            print(f"Distâncias até o centro da lane:")
            print(f"  Média: {np.mean(distances):.3f} m")
            print(f"  Desvio padrão: {np.std(distances):.3f} m")
            print(f"  Mínimo: {np.min(distances):.3f} m")
            print(f"  Máximo: {np.max(distances):.3f} m")
        
        if 'pointclouds' in data:
            pointclouds = data['pointclouds']
            print("\nPointclouds:")
            print(f"  Número total: {len(pointclouds)}")
            print(f"  Formato de cada pointcloud: {pointclouds[0].shape}")
            print(f"  Tipo de dados: {pointclouds[0].dtype}")
        
        # 4. Mostra amostras específicas
        print(f"\n[4/4] AMOSTRAS (mostrando {num_samples}):")
        indices = np.random.choice(len(data['pointclouds']), 
                                 size=min(num_samples, len(data['pointclouds'])), 
                                 replace=False)
        
        for i, idx in enumerate(indices):
            print(f"\nAmostra {i+1} (índice {idx}):")
            print(f"Arquivo de origem: {data['source_files'][idx]}")
            print(f"Timestamp global: {data['global_timestamps'][idx]:.6f}")
            print(f"Timestamp relativo: {data['relative_timestamps'][idx]:.2f} s")
            print(f"Posição (x,y,θ): ({data['x'][idx]:.2f}, {data['y'][idx]:.2f}, {data['theta'][idx]:.3f} rad)")
            print(f"Distância até centro: {data['distances'][idx]:.3f} m")
            print(f"Pointcloud (primeiros 3 pontos):")
            print(data['pointclouds'][idx][:3])  # Mostra apenas os 3 primeiros pontos
            
    except Exception as e:
        print(f"\nErro ao visualizar dados: {str(e)}")
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
    print(split_params)
    """
    
    train_data, test_data = sample_pointclouds(data_dir, split_line=split_params['split_line'], test_side=split_params['test_side'])

    # Calcula distâncias para ambos os conjuntos
    train_data = calculate_distances(train_data, median_path)
    test_data = calculate_distances(test_data, median_path)

    save_training_data(train_data, test_data, output_dir)
    view_training_data(output_dir + '/complete_training_data.npz')
    """
    
    test_points  = [
        {
            'pointcloud': np.random.rand(100, 3),  # Dummy pointcloud
            'globalpos': np.array([median_path['centroids'][0][0], median_path['centroids'][0][1], 0.0]),  # [x, y, theta]
            'global_timestamp': 123456.789,
            'relative_timestamp': 0.0,
            'source_file': 'test_file1.npy',
            'original_index': 0
        },
        {
            'pointcloud': np.random.rand(100, 3),
            'globalpos': np.array([757200, -363800, 0.1]),
            'global_timestamp': 123457.789,
            'relative_timestamp': 1.0,
            'source_file': 'test_file2.npy',
            'original_index': 1
        },
        {
            'pointcloud': np.random.rand(100, 3),
            'globalpos': np.array([756800, -364000, -0.1]),
            'global_timestamp': 123458.789,
            'relative_timestamp': 2.0,
            'source_file': 'test_file3.npy',
            'original_index': 2
        },
        {
            'pointcloud': np.random.rand(100, 3),
            'globalpos': np.array([756800, -363700, 0.2]),
            'global_timestamp': 123459.789,
            'relative_timestamp': 3.0,
            'source_file': 'test_file4.npy',
            'original_index': 3
        },
        {
            'pointcloud': np.random.rand(100, 3),
            'globalpos': np.array([757600, -363900, -0.2]),
            'global_timestamp': 123460.789,
            'relative_timestamp': 4.0,
            'source_file': 'test_file5.npy',
            'original_index': 4
        },
        {
            'pointcloud': np.random.rand(100, 3),
            'globalpos': np.array([757800, -363700, 0.3]),
            'global_timestamp': 123461.789,
            'relative_timestamp': 5.0,
            'source_file': 'test_file6.npy',
            'original_index': 5
        }
    ]

    # Carrega os dados completos de treinamento
    training_data_path = os.path.join(output_dir, "complete_training_data.npz")
    training_data = np.load(training_data_path, allow_pickle=True)
    
    # Separa índices de treino e teste
    train_indices = np.where(training_data['train_mask'])[0]
    test_indices = np.where(~training_data['train_mask'].astype(bool))[0]

    # Seleciona 5 pontos aleatórios de cada conjunto
    np.random.seed(42)  # Para reprodutibilidade
    selected_train = np.random.choice(train_indices, size=min(5, len(train_indices)), replace=False)
    selected_test = np.random.choice(test_indices, size=min(5, len(test_indices)), replace=False)

    # Prepara a estrutura para plotagem
    points_with_distances = []
    
    # Adiciona pontos de treino selecionados
    for idx in selected_train:
        points_with_distances.append({
            'point': np.array([training_data['x'][idx], training_data['y'][idx]]),
            'distance': training_data['distances'][idx],
            'path_segment': training_data.get('path_segments', [0]*len(training_data['x']))[idx],
            'is_test': False
        })
    
    # Adiciona pontos de teste selecionados
    for idx in selected_test:
        points_with_distances.append({
            'point': np.array([training_data['x'][idx], training_data['y'][idx]]),
            'distance': training_data['distances'][idx],
            'path_segment': training_data.get('path_segments', [0]*len(training_data['x']))[idx],
            'is_test': True
        })
    
    # Embaralha os pontos para misturar treino/teste no plot
    np.random.shuffle(points_with_distances)
    
    # Gera o gráfico
    print("\nGerando gráfico de distâncias com amostra aleatória...")
    plot_with_distances(
        points_with_distances=points_with_distances,
        median_path=median_path,
        split_line=split_params['split_line'] if split_params else None,
        test_side=split_params['test_side'] if split_params else 'left',
        plot_dir=output_dir,
        max_points_to_plot=10  # Plotará todos os 10 pontos selecionados
    )
    
    print("\nProcesso concluído! Gráfico salvo em:", os.path.join(output_dir, 'distancias_amostra_aleatoria.png'))

    print("\nProcesso concluído com sucesso!")

if __name__ == "__main__":
    main()