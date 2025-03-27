import numpy as np
import os
from scipy.spatial import KDTree
from collections import defaultdict
from scipy.signal import savgol_filter

# Constantes
VELODYNE_TAG = 'VELODYNE_PARTIAL_SCAN_IN_FILE'
TIME_WINDOW = 0.2  # Janela temporal para agrupamento (em segundos)
SAMPLES_PER_FILE = 200

def load_all_globalpos(data_dir):
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

def calculate_median_path(all_globalpos):
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
        median_points = smooth_path(median_points)
    
    print(f"Caminho mediano calculado com {len(median_points)} pontos")
    return median_points

def smooth_path(path, window_size=5, polyorder=2):
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

def sample_pointclouds(data_dir):
    """Amostra pointclouds e associa ao caminho mediano"""
    print("\n[3/4] Amostrando pointclouds...")
    sampled_data = []
    
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
                    
                    sampled_data.append({
                        'pointcloud': item['pointcloud'],
                        'globalpos': np.array([item['x'], item['y'], item['theta']]),
                        'global_timestamp': item['timestamp'],
                        'relative_timestamp': rel_time,
                        'source_file': filename,
                        'original_index': idx
                    })
                    
        except Exception as e:
            print(f"Erro ao processar {filename}: {str(e)}")
    
    print(f"Total de pointclouds amostradas: {len(sampled_data)}")
    return sampled_data

def calculate_distances(sampled_data, median_path):
    """Calcula distâncias até o caminho mediano considerando theta"""
    print("\n[4/5] Calculando distâncias...")
    
    # Prepara KDTree com posições do caminho mediano
    median_coords = np.array([[p['x'], p['y']] for p in median_path])
    tree = KDTree(median_coords)
    
    for item in sampled_data:
        # Acessa as coordenadas diretamente do item (estrutura fixa)
        point = np.array([item['globalpos'][0], item['globalpos'][1]])  # x, y
        theta = item['globalpos'][2]  # theta
        
        # Encontra os 2 pontos mais próximos
        _, indices = tree.query(point, k=2)
        closest_indices = sorted(indices)
        
        if len(closest_indices) < 2:
            item['distance'] = 0.0
            continue
            
        # Pega segmento mais próximo
        p1 = median_path[closest_indices[0]]
        p2 = median_path[closest_indices[1]]
        
        # Vetores para cálculo
        seg_vec = np.array([p2['x']-p1['x'], p2['y']-p1['y']])
        point_vec = np.array([point[0]-p1['x'], point[1]-p1['y']])
        
        # Evita divisão por zero
        seg_norm = np.dot(seg_vec, seg_vec)
        if seg_norm < 1e-6:
            item['distance'] = np.linalg.norm(point_vec)
            continue
            
        # Projeção no segmento
        t = np.dot(point_vec, seg_vec) / seg_norm
        t = np.clip(t, 0, 1)
        proj_point = p1['x'] + t*seg_vec[0], p1['y'] + t*seg_vec[1]
        
        # Distância com sinal (considerando orientação)
        cross = seg_vec[0]*point_vec[1] - seg_vec[1]*point_vec[0]
        distance = np.linalg.norm(point - proj_point)
        signed_dist = np.sign(cross) * distance
        
        # Ponderação angular
        theta_diff = min(abs(theta-p1['theta']), abs(theta-p2['theta']))
        angular_weight = np.cos(theta_diff)
        
        item['distance'] = signed_dist * angular_weight
        item['path_index'] = closest_indices[0]
    
    return sampled_data

def save_training_data(sampled_data, output_dir):
    """Salva todos os dados de treinamento em um único arquivo .npz formatado"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n[5/5] Consolidando e salvando dados de treinamento...")
    
    # Extrai todos os dados em arrays separados
    x = np.array([item['globalpos'][0] for item in sampled_data], dtype=np.float32)
    y = np.array([item['globalpos'][1] for item in sampled_data], dtype=np.float32)
    theta = np.array([item['globalpos'][2] for item in sampled_data], dtype=np.float32)
    global_timestamps = np.array([item['global_timestamp'] for item in sampled_data], dtype=np.float64)
    relative_timestamps = np.array([item['relative_timestamp'] for item in sampled_data], dtype=np.float64)
    distances = np.array([item['distance'] for item in sampled_data], dtype=np.float32)
    pointclouds = np.array([item['pointcloud'] for item in sampled_data], dtype=object)
    source_files = np.array([item['source_file'] for item in sampled_data], dtype=object)
    
    # Cria dicionário com metadados
    metadata = {
        'description': 'Dataset de treinamento para detecção de faixa',
        'num_samples': len(sampled_data),
        'variables': ['x', 'y', 'theta', 'global_timestamp', 'relative_timestamp', 
                     'distance', 'pointcloud', 'source_file'],
        'units': {
            'x': 'metros',
            'y': 'metros',
            'theta': 'radianos',
            'timestamps': 'segundos',
            'distance': 'metros'
        }
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
        metadata=metadata
    )
    
    print(f"Dados salvos em: {output_path}")
    print(f"Total de amostras: {len(sampled_data)}")
    print("Variáveis incluídas:")
    print("- Posições (x, y, theta)")
    print("- Timestamps (global e relativo)")
    print("- Distâncias até o centro da faixa")
    print("- Pointclouds completas")
    print("- Arquivos de origem")

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
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "output")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "training_data")
    
    # Pipeline principal
    globalpos_data = load_all_globalpos(data_dir)
    median_path = calculate_median_path(globalpos_data)
    sampled_data = sample_pointclouds(data_dir)
    sampled_data = calculate_distances(sampled_data, median_path)
    save_training_data(sampled_data, output_dir)
    view_training_data(output_dir + '/complete_training_data.npz')
    
    print("\nProcesso concluído com sucesso!")

if __name__ == "__main__":
    main()