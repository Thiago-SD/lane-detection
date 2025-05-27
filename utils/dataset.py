import os
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from scipy.interpolate import splprep, splev
from pathlib import Path


# Ângulos verticais do Velodyne
velodyne_vertical_angles = np.array([
    -30.6700000, -29.3300000, -28.0000000, -26.6700000, -25.3300000, -24.0000000, -22.6700000, -21.3300000,
    -20.0000000, -18.6700000, -17.3300000, -16.0000000, -14.6700000, -13.3300000, -12.0000000, -10.6700000,
    -9.3299999, -8.0000000, -6.6700001, -5.3299999, -4.0000000, -2.6700001, -1.3300000, 0.0000000, 1.3300000,
    2.6700001, 4.0000000, 5.3299999, 6.6700001, 8.0000000, 9.3299999, 10.6700000
])

# Ordem dos raios do Velodyne
velodyne_ray_order = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31])

# Constantes
N_RAYS = 1084  # Número fixo de raios por nuvem de pontos

# Constantes para as tags
GPS_TAG = 'NMEAGGA'
GPS_ORIENTATION_TAG = 'NMEAHDT'
ODOM_TAG = 'ROBOTVELOCITY_ACK'
CAMERA_TAG = 'BUMBLEBEE_BASIC_STEREOIMAGE_IN_FILE3'
VELODYNE_TAG = 'VELODYNE_PARTIAL_SCAN_IN_FILE'
XSENS_TAG = 'XSENS_QUAT'

# Função para ler e converter dados binários em pontos 3D
def binaryTo3d(file_path):
    # Verificar se o arquivo existe e não está vazio
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"Arquivo vazio: {file_path}")

    # Tamanho dos tipos de dados
    size_double = np.dtype(np.float64).itemsize  # 8 bytes
    size_short = np.dtype(np.int16).itemsize  # 2 bytes
    size_char = np.dtype(np.uint8).itemsize  # 1 byte

    # Tamanho de cada registro (disparo)
    record_size = size_double + (32 * size_short) + (32 * size_char)

    # Verificar o tamanho do arquivo
    file_size = os.path.getsize(file_path)
    expected_size = N_RAYS * record_size

    # Se o arquivo for menor que o esperado, ajustar o número de raios
    if file_size < expected_size:
        actual_rays = file_size // record_size
        print(f"Aviso: Arquivo {file_path} tem tamanho inesperado. Processando {actual_rays} raios em vez de {N_RAYS}.")
    else:
        actual_rays = N_RAYS

    # Inicializar array para armazenar os pontos 3D
    points = np.empty((actual_rays * 32, 3), dtype=np.float32)  # (N, 3) onde N = actual_rays * 32

    try:
        with open(file_path, 'rb') as f:
            for i in range(actual_rays):
                # Ler o ângulo (double, 8 bytes)
                angle_data = f.read(size_double)
                if len(angle_data) != size_double:
                    print(f"Aviso: Erro ao ler o ângulo no disparo {i}. Pulando este disparo.")
                    continue
                angle = np.frombuffer(angle_data, dtype=np.float64)[0]

                # Ler as distâncias (32 * short, 2 bytes cada)
                distances_data = f.read(32 * size_short)
                if len(distances_data) != 32 * size_short:
                    print(f"Aviso: Erro ao ler distâncias no disparo {i}. Pulando este disparo.")
                    continue
                distances = np.frombuffer(distances_data, dtype=np.int16)

                # Ler as intensidades (32 * char, 1 byte cada)
                intensities_data = f.read(32 * size_char)
                if len(intensities_data) != 32 * size_char:
                    print(f"Aviso: Erro ao ler intensidades no disparo {i}. Pulando este disparo.")
                    continue
                intensities = np.frombuffer(intensities_data, dtype=np.uint8)

                # Converter o ângulo para radianos
                h_angle = np.pi * angle / 180.0

                # Processar cada ponto do disparo
                for j in range(32):
                    l_range = distances[j] / 500.0  # Normalizar distância
                    v_angle = velodyne_vertical_angles[j]
                    v_angle = np.pi * v_angle / 180.0  # Converter ângulo vertical para radianos

                    # Calcular as coordenadas x, y, z
                    cos_rot_angle = np.cos(h_angle)
                    sin_rot_angle = np.sin(h_angle)
                    cos_vert_angle = np.cos(v_angle)
                    sin_vert_angle = np.sin(v_angle)

                    xy_distance = l_range * cos_vert_angle
                    x = xy_distance * cos_rot_angle
                    y = xy_distance * sin_rot_angle
                    z = l_range * sin_vert_angle

                    # Armazenar o ponto no array
                    points[i * 32 + j, 0] = x
                    points[i * 32 + j, 1] = y
                    points[i * 32 + j, 2] = z

    except Exception as e:
        print(f"Erro ao processar o arquivo {file_path}: {e}")
        return None

    return points

# Função para encontrar a mensagem mais próxima no tempo
def find_nearest_timestamp(target_timestamp, timestamps):
    """
    Encontra o índice da mensagem mais próxima no tempo para um determinado timestamp.
    :param target_timestamp: Timestamp de referência.
    :param timestamps: Array de timestamps.
    :return: Índice da mensagem mais próxima.
    """
    diffs = np.abs(timestamps - target_timestamp)
    #print(target_timestamp)
    return np.argmin(diffs)

def read_globalpos_file(globalpos_file_path):
    """
    Lê o arquivo de posição global e retorna os dados como um array NumPy estruturado.
    :param globalpos_file_path: Caminho para o arquivo de posição global.
    :return: Array NumPy estruturado com campos x, y, theta e timestamp.
    """
    globalpos_data = []
    with open(globalpos_file_path, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip() == "msg_begin":
                x = float(lines[i + 1].strip())
                y = float(lines[i + 2].strip())
                theta = float(lines[i + 3].strip())
                timestamp = float(lines[i + 10].strip())
                #print(timestamp)
                globalpos_data.append((x, y, theta, timestamp))
                i += 10
            else:
                i += 1

    return np.array(globalpos_data, dtype=np.dtype([
        ('x', np.float64),
        ('y', np.float64),
        ('theta', np.float64),
        ('timestamp', np.float64)
    ]))

def load_all_globalpos(data_dir):
    """Carrega todas as posições globais dos arquivos .txt de globalpos"""
    print("\nCarregando dados de posição global dos arquivos txt...")
    
    # Lista todos os arquivos globalpos_*.txt no diretório
    all_files = sorted([f for f in os.listdir(data_dir) 
                       if f.startswith('globalpos_') and f.endswith('.txt')])
    
    # Array para armazenar os arrays de cada arquivo
    all_data_arrays = []
    
    for filename in all_files:
        filepath = os.path.join(data_dir, filename)
        print(f"Processando arquivo: {filename}")
        
        try:
            # Usa a função do dataset.py para ler o arquivo
            file_data = read_globalpos_file(filepath)
            
            if len(file_data) == 0:
                print(f"Arquivo sem dados válidos: {filename}")
                continue
            
            # Calcula timestamp relativo (baseado no primeiro timestamp do arquivo)
            base_timestamp = file_data['timestamp'][0]
            relative_timestamps = file_data['timestamp'] - base_timestamp
            
            # Converte para o formato desejado (array de dicionários)
            file_data_list = []
            for i in range(len(file_data)):
                file_data_list.append({
                    'x': file_data['x'][i],
                    'y': file_data['y'][i],
                    'theta': file_data['theta'][i],
                    'global_timestamp': file_data['timestamp'][i],
                    'relative_timestamp': relative_timestamps[i],
                    'filename': filename,
                    'file_index': i
                })
            
            # Adiciona o array deste arquivo à lista principal
            all_data_arrays.append(np.array(file_data_list))
            
        except Exception as e:
            print(f"Erro ao processar {filename}: {str(e)}")
            continue
    
    print(f"Total de arquivos processados: {len(all_data_arrays)}")
    print(f"Total de pontos carregados: {sum(len(arr) for arr in all_data_arrays)}")
    
    return all_data_arrays

def calculate_median_path(all_data_arrays, plot_dir=None, n_clusters=100, n_segments=5):
    """Calcula caminho mediano dividindo em segmentos e interpolando cada um separadamente."""
    # Concatena e processa os dados
    all_data = np.concatenate(all_data_arrays)
    points = np.array([[item['x'], item['y']] for item in all_data])
    thetas = np.array([item['theta'] for item in all_data])
    
    # 1. Clusterização inicial com K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(points)
    centroids = kmeans.cluster_centers_
    
    # 2. Divisão dos centroides em segmentos
    # Ordena os centroides por ângulo polar em relação ao centroide médio
    mean_point = centroids.mean(axis=0)
    angles = np.arctan2(centroids[:,1]-mean_point[1], centroids[:,0]-mean_point[0])
    sorted_indices = np.argsort(angles)
    sorted_centroids = centroids[sorted_indices]
    
    # Divide em n_segments partes aproximadamente iguais
    segment_indices = np.array_split(sorted_indices, n_segments)
    
    # 3. Processa cada segmento separadamente
    all_x_interp = []
    all_y_interp = []
    all_theta_interp = []
    segment_labels = []  # Para armazenar os rótulos de segmento
    
    for seg_num, segment_idx in enumerate(segment_indices):
        segment_centroids = centroids[segment_idx]
        
        if len(segment_centroids) < 3:  # Tratamento para segmentos pequenos
            all_x_interp.extend(segment_centroids[:,0])
            all_y_interp.extend(segment_centroids[:,1])
            if len(segment_centroids) > 1:
                theta = np.arctan2(segment_centroids[-1,1]-segment_centroids[0,1],
                                  segment_centroids[-1,0]-segment_centroids[0,0])
                all_theta_interp.extend([theta]*len(segment_centroids))
            else:
                all_theta_interp.extend([0]*len(segment_centroids))
            segment_labels.extend([seg_num]*len(segment_centroids))
            continue
            
        # Parametrização por distância acumulada
        diffs = np.diff(segment_centroids, axis=0)
        dists = np.sqrt((diffs**2).sum(axis=1))
        t = np.concatenate([[0], np.cumsum(dists)])
        
        # Interpolação spline
        tck, _ = splprep([segment_centroids[:,0], segment_centroids[:,1]], u=t, s=0)
        
        # Gera pontos interpolados (número proporcional ao tamanho do segmento)
        n_points = max(50, len(segment_centroids)*2)
        u_new = np.linspace(0, 1, n_points)
        x_interp, y_interp = splev(u_new, tck)
        dx, dy = splev(u_new, tck, der=1)
        theta_interp = np.arctan2(dy, dx)
        
        all_x_interp.extend(x_interp)
        all_y_interp.extend(y_interp)
        all_theta_interp.extend(theta_interp)
        segment_labels.extend([seg_num]*len(x_interp))
    
    # 4. Combina todos os segmentos
    combined_path = [{
        'x': all_x_interp[i],
        'y': all_y_interp[i],
        'theta': all_theta_interp[i],
        'relative_timestamp': float(i)/len(all_x_interp),
        'n_points': len(points),
        'segment': segment_labels[i]  # Usa os rótulos pré-calculados
    } for i in range(len(all_x_interp))]
    
    # 5. Visualização
    if plot_dir:
        plot_segmented_interpolation(points, centroids, segment_indices, 
                                   all_x_interp, all_y_interp, plot_dir)
    
    return combined_path

def plot_segmented_interpolation(points, centroids, segment_indices, 
                               x_interp, y_interp, plot_dir):
    """Visualização do processo com segmentação"""
    try:
        plt.figure(figsize=(15, 10))
        colors = plt.cm.tab10.colors  # Paleta de cores para os segmentos
        
        # Plot 1: Pontos originais e clusters
        plt.subplot(2, 2, 1)
        plt.scatter(points[:,0], points[:,1], c='gray', alpha=0.1, s=5, label='Pontos originais')
        plt.scatter(centroids[:,0], centroids[:,1], c='red', s=30, label='Centroides K-Means')
        plt.title('Agrupamento Inicial')
        plt.legend()
        
        # Plot 2: Segmentação dos centroides
        plt.subplot(2, 2, 2)
        for i, seg_idx in enumerate(segment_indices):
            seg_centroids = centroids[seg_idx]
            plt.scatter(seg_centroids[:,0], seg_centroids[:,1], 
                       color=colors[i % len(colors)], 
                       label=f'Segmento {i+1}')
        plt.title('Centroides Segmentados')
        plt.legend()
        
        # Plot 3: Trajetória final interpolada
        plt.subplot(2, 2, 3)
        plt.scatter(points[:,0], points[:,1], c='gray', alpha=0.05, s=5)
        
        # Plota cada segmento com cor diferente
        n_total = len(x_interp)
        n_segments = len(segment_indices)
        seg_length = n_total // n_segments
        
        for i in range(n_segments):
            start = i * seg_length
            end = (i+1) * seg_length if i < n_segments-1 else n_total
            plt.plot(x_interp[start:end], y_interp[start:end], 
                    color=colors[i % len(colors)], 
                    linewidth=2, label=f'Segmento {i+1}')
        
        plt.title('Trajetória Final Interpolada')
        plt.legend()
        
        # Plot 4: Todos os elementos juntos
        plt.subplot(2, 2, 4)
        plt.scatter(points[:,0], points[:,1], c='gray', alpha=0.02, s=5)
        
        # Centroides coloridos por segmento
        for i, seg_idx in enumerate(segment_indices):
            seg_centroids = centroids[seg_idx]
            plt.scatter(seg_centroids[:,0], seg_centroids[:,1], 
                       color=colors[i % len(colors)], s=30)
        
        # Trajetória final
        plt.plot(x_interp, y_interp, 'k-', linewidth=1.5, alpha=0.7)
        
        plt.title('Visão Geral')
        
        plt.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f"{plot_dir}/segmented_path.png", dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Erro ao gerar gráficos: {str(e)}")

def plot_individual_routes(data_dir, output_dir):
    """
    Gera gráficos individuais para cada arquivo globalpos
    """
    # Encontra todos os arquivos globalpos
    files = sorted(Path(data_dir).glob('globalpos_*.txt'))
    
    if not files:
        print("Nenhum arquivo globalpos_*.txt encontrado no diretório.")
        return
    
    # Cria diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot individual para cada arquivo
    for filepath in files:
        points = read_globalpos_file(filepath)
        n_points = len(points)
        
        plt.figure(figsize=(10, 8))
        plt.plot(points['x'], points['y'], 'b-', linewidth=1, label='Trajetória')
        plt.scatter(points['x'][0], points['y'][0], c='green', s=100, label='Início')
        plt.scatter(points['x'][n_points - 1], points['y'][n_points - 1], c='red', s=100, label='Fim')
        
        plt.title(f'Rota: {filepath.name}')
        plt.xlabel('Coordenada X')
        plt.ylabel('Coordenada Y')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # Salva o gráfico
        output_path = os.path.join(output_dir, f'{filepath.stem}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'Gráfico salvo: {output_path}')

def print_dataset_structure(dataset_path, num_samples=5):
    """
    Carrega o dataset numpy e imprime o formato dos vetores armazenados.
    :param dataset_path: Caminho para o arquivo .npy
    :param num_samples: Número de entradas aleatórias a serem exibidas (padrão: 5)
    """
    try:
        # Carregar o dataset
        dataset = np.load(dataset_path, allow_pickle=True)

        # Verificar se o dataset foi carregado corretamente
        if not isinstance(dataset, np.ndarray):
            print(f"Erro: O arquivo {dataset_path} não contém um array numpy válido.")
            return

        # Imprimir o número de entradas no dataset
        print(f"O dataset contém {len(dataset)} entradas.")

        # Selecionar entradas aleatórias do dataset
        if len(dataset) > 0:
            random_indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
            for i, idx in enumerate(random_indices):
                entry = dataset[idx]
                print(f"\nEntrada {i + 1} (índice {idx}):")
                print(f"  - Timestamp: {entry['timestamp']}")
                print(f"  - Posição global (x, y, theta): ({entry['x']}, {entry['y']}, {entry['theta']})")
                print(f"  - Nuvem de pontos (pointcloud):")
                print(f"    - Formato: {entry['pointcloud'].shape}")
                print(f"    - Tipo de dados: {entry['pointcloud'].dtype}")
                print(f"    - Primeiros 5 pontos:")
                print(entry['pointcloud'][:5])  # Imprimir os primeiros 5 pontos
        else:
            print("O dataset está vazio.")
    except Exception as e:
        print(f"Erro ao carregar ou processar o arquivo {dataset_path}: {e}")

def main():
    # Obter o caminho absoluto do diretório do script
    script_path = os.path.dirname(os.path.abspath(__file__))

    # Caminho para o diretório de dados de entrada
    input_path = os.path.abspath(os.path.join(script_path, "..", "data", "input"))

    # Caminho para o diretório de saída
    output_dir = os.path.abspath(os.path.join(script_path, "..", "data", "output"))
    os.makedirs(output_dir, exist_ok=True)

    # Verificar se o diretório de entrada existe
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_path}")
    
    # Listar diretórios no primeiro nível de /input
    try:
        dirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d)) and d.startswith("log_volta_da_ufes_")]
    except Exception as e:
        print(f"Erro ao listar diretórios em {input_path}: {e}")
        return
    
    #plot_individual_routes(input_path, output_dir)
    #return

    # Calcular o Caminho mediano
    globalpos_data = load_all_globalpos(input_path)
    median_path = calculate_median_path(globalpos_data, plot_dir=os.path.join(output_dir, "caminho_mediano"), n_clusters=100, n_segments=10)

    # Salvar o Caminho mediano para uso no pré-processamento
    np.save(os.path.join(output_dir, "caminho_mediano") + '/caminho_mediano.npy', median_path)

    return

    # Processar cada diretório de log
    for dir_name in dirs:
        log_dir_path = os.path.join(input_path, dir_name)
        print(f"Processando diretório de log: {log_dir_path}")

        # Caminho para o arquivo de log
        log_file_path = os.path.join(log_dir_path, f"{dir_name}.txt")
        if not os.path.exists(log_file_path):
            print(f"Arquivo de log não encontrado: {log_file_path}")
            continue

        # Caminho para o arquivo de posição global correspondente
        globalpos_file_path = os.path.join(input_path, f"globalpos_{dir_name}.txt")
        if not os.path.exists(globalpos_file_path):
            print(f"Arquivo de posição global não encontrado: {globalpos_file_path}")
            continue

        # Caminho para o diretório de nuvens de pontos
        pointcloud_dir = os.path.join(log_dir_path, f"{dir_name}.txt_lidar")
        if not os.path.exists(pointcloud_dir):
            print(f"Diretório de nuvens de pontos não encontrado: {pointcloud_dir}")
            continue

        # Caminho para o arquivo .npy de saída
        dataset_output_path = os.path.join(output_dir, f"{dir_name}.npy")

        try:
            # Ler o arquivo de posição global
            globalpos_data = read_globalpos_file(globalpos_file_path)
            if len(globalpos_data) == 0:
                print(f"Erro: Nenhum dado de posição global encontrado no arquivo {globalpos_file_path}.")
                continue

            # Arrays NumPy para armazenar nuvens de pontos
            pointcloud_timestamps = np.array([], dtype=np.float64)
            pointcloud_data = []  # Lista temporária para nuvens de pontos

            # Lista para armazenar o dataset
            dataset_np = []

            # Abrir o arquivo de log
            with open(log_file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2 or line.startswith('#'):
                        continue

                    elif parts[0] == VELODYNE_TAG:  # Nuvens de pontos (VELODYNE_PARTIAL_SCAN_IN_FILE)
                        try:
                            # Verificar se a mensagem tem o número correto de partes
                            if len(parts) < 6:
                                print(f"Mensagem de nuvem de pontos malformada: {line}")
                                continue

                            # Extrair o caminho do arquivo .pointcloud
                            pc_file = parts[1]
                            if pc_file.startswith("_lidar/"):
                                pc_file = pc_file[len("_lidar/"):]

                            # Construir o caminho completo do arquivo .pointcloud
                            pc_path = os.path.join(pointcloud_dir, pc_file)

                            # Verificar se o arquivo .pointcloud existe
                            if not os.path.exists(pc_path):
                                print(f"Arquivo .pointcloud não encontrado: {pc_path}")
                                continue

                            # Converter dados binários em pontos 3D
                            points = binaryTo3d(pc_path)
                            if points is None:
                                print(f"Erro ao converter nuvem de pontos: {pc_path}")
                                continue

                            # Extrair o timestamp da nuvem de pontos
                            velodyne_timestamp = float(parts[-3])

                            # Armazenar nuvem de pontos no array
                            pointcloud_timestamps = np.append(pointcloud_timestamps, velodyne_timestamp)
                            pointcloud_data.append(points)

                            #print(f"Nuvem de pontos processada: timestamp={velodyne_timestamp}, arquivo={pc_path}")
                        except (IndexError, ValueError) as e:
                            print(f"Erro ao processar mensagem de nuvem de pontos: {line}. Detalhes: {e}")
                            continue

            # Associar nuvens de pontos a posições globais
            print("Associar nuvens de pontos a posições globais")
            for i, pc_timestamp in enumerate(pointcloud_timestamps):
                try:
                    # Encontrar a posição global mais próxima no tempo
                    nearest_index = find_nearest_timestamp(pc_timestamp, globalpos_data['timestamp'])
                    
                    # Verificar se nearest_index é válido
                    if nearest_index < 0 or nearest_index >= len(globalpos_data):
                        print(f"Índice inválido para globalpos_data: {nearest_index}")
                        continue

                    # Acessar a posição global mais próxima
                    nearest_position = globalpos_data[nearest_index]

                    # Adicionar ao dataset numpy
                    dataset_np.append({
                        "timestamp": pc_timestamp,
                        "x": nearest_position['x'],
                        "y": nearest_position['y'],
                        "theta": nearest_position['theta'],
                        "pointcloud": pointcloud_data[i]
                    })
                except Exception as e:
                    print(f"Erro ao associar nuvem de pontos a posição global: {e}")
                    continue

            # Salvar o dataset numpy
            np.save(dataset_output_path, dataset_np)
            print(f"Dataset salvo em {dataset_output_path}")
            print_dataset_structure(dataset_output_path)

        except FileNotFoundError as e:
            print(f"Erro: {e}")
        except Exception as e:
            print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()