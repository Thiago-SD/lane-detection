import os
import numpy as np
import math

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

    # Verificar se o diretório de entrada existe
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Diretório de entrada não encontrado: {input_path}")
    
    # Listar diretórios no primeiro nível de /input
    try:
        dirs = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d)) and d.startswith("log_volta_da_ufes_")]
    except Exception as e:
        print(f"Erro ao listar diretórios em {input_path}: {e}")
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

        # Caminho para o diretório de saída
        output_dir = os.path.abspath(os.path.join(script_path, "..", "data", "output"))
        os.makedirs(output_dir, exist_ok=True)

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