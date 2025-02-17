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

# Classe para manter o estado do robô (posição e orientação)
class RobotState:
    def __init__(self):
        self.x = 0.0  # Posição global X
        self.y = 0.0  # Posição global Y
        self.theta = 0.0  # Orientação (em radianos)
        self.last_timestamp = None  # Último timestamp processado

    def update(self, v, phi, timestamp):
        """
        Atualiza a posição e orientação do robô com base na velocidade linear (v) e angular (phi).
        :param v: Velocidade linear (m/s)
        :param phi: Velocidade angular (rad/s)
        :param timestamp: Timestamp atual (segundos)
        """
        if self.last_timestamp is not None:
            # Calcular o tempo decorrido desde a última atualização
            dt = timestamp - self.last_timestamp

            # Atualizar a orientação (theta)
            self.theta += phi * dt

            # Atualizar a posição (x, y)
            self.x += v * math.cos(self.theta) * dt
            self.y += v * math.sin(self.theta) * dt

        # Atualizar o último timestamp
        self.last_timestamp = timestamp

    def get_position(self):
        """
        Retorna a posição e orientação atual do robô.
        :return: (x, y, theta)
        """
        return self.x, self.y, self.theta

# Função para encontrar a mensagem mais próxima no tempo
def find_near(s, queue, key):
    near = 0
    for i in range(1, len(queue[key])):
        dti = abs(float(queue[key][i][0]) - float(s[-1]))  # Comparar timestamps
        dtn = abs(float(queue[key][near][0]) - float(s[-1]))
        if dti < dtn:
            near = i
    return near

# Função para adicionar dados à fila
def add_to_queue(queue, key, data):
    if key not in queue:
        queue[key] = []
    queue[key].append(data)

# Função para parsear o arquivo de log
def parse_log_file(log_file_path):
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Arquivo de log não encontrado: {log_file_path}")

    queue = {
        'velodyne': [],  # Nuvens de pontos
        'odom': [],      # Odometria (ROBOTVELOCITY_ACK)
    }

    robot_state = RobotState()  # Inicializar o estado do robô

    with open(log_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            if parts[0] == "VELODYNE_PARTIAL_SCAN_IN_FILE":
                add_to_queue(queue, 'velodyne', parts)
            elif parts[0] == "ROBOTVELOCITY_ACK":
                try:
                    v = float(parts[1])  # Velocidade linear (m/s)
                    phi = float(parts[2])  # Velocidade angular (rad/s)
                    timestamp = float(parts[3])  # Timestamp
                    robot_state.update(v, phi, timestamp)
                    #print(robot_state.get_position())
                    add_to_queue(queue, 'odom', [timestamp, *robot_state.get_position()])
                except (IndexError, ValueError) as e:
                    print(f"Erro ao processar mensagem de odometria: {line}. Detalhes: {e}")
                    continue

    return queue

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

# Função para associar nuvens de pontos a posições globais
def associate_pointclouds_with_positions(queue):
    dataset = []

    for velodyne_data in queue['velodyne']:
        velodyne_timestamp = float(velodyne_data[-1])

        # Encontrar a mensagem de odometria mais próxima no tempo
        nearest_odom_index = find_near(velodyne_data, queue, 'odom')
        nearest_odom = queue['odom'][nearest_odom_index]

        # Extrair a posição global
        timestamp, x, y, theta = nearest_odom

        # Adicionar ao dataset
        dataset.append((velodyne_data[1], (timestamp, x, y, theta)))

    return dataset

def print_dataset_structure(dataset_path):
    """
    Carrega o dataset numpy e imprime o formato dos vetores armazenados.
    :param dataset_path: Caminho para o arquivo .npy
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

        # Iterar sobre as entradas do dataset
        for i, entry in enumerate(dataset):
            print(f"\nEntrada {i + 1}:")
            print(f"  - Timestamp: {entry['timestamp']}")
            print(f"  - Posição global (x, y, theta): ({entry['x']}, {entry['y']}, {entry['theta']})")
            print(f"  - Nuvem de pontos (pointcloud):")
            print(f"    - Formato: {entry['pointcloud'].shape}")
            print(f"    - Tipo de dados: {entry['pointcloud'].dtype}")
            print(f"    - Primeiros 5 pontos:")
            print(entry['pointcloud'][:5])  # Imprimir os primeiros 5 pontos

    except Exception as e:
        print(f"Erro ao carregar ou processar o arquivo {dataset_path}: {e}")

def main():
    # Obter o caminho absoluto do diretório do script
    script_path = os.path.dirname(os.path.abspath(__file__))

    # Caminho para o diretório de dados
    data_path = os.path.abspath(os.path.join(script_path, "..", "data"))

    # Verificar se o diretório de dados existe
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Diretório de dados não encontrado: {data_path}")

    # Percorrer todos os subdiretórios em /data
    for root, dirs, files in os.walk(data_path):
        for file in files:
            # Verificar se o arquivo é um log relevante (por exemplo, log_volta_da_ufes_*.txt)
            if file.startswith("log_volta_da_ufes_") and file.endswith(".txt"):
                log_file_path = os.path.join(root, file)
                print(f"Processando arquivo de log: {log_file_path}")

                # Extrair o nome do diretório raiz (ex: log_volta_da_ufes_20230422)
                root_dir_name = os.path.basename(root)
                # Caminho para o diretório de nuvens de pontos (log_volta_da_ufes_20230422.txt_lidar)
                pointcloud_dir = os.path.join(root, f"{root_dir_name}.txt_lidar")
                # Caminho para o diretório de saída
                output_dir = os.path.join(data_path, "output")
                # Verificar se o diretório de saída existe
                os.makedirs(output_dir, exist_ok=True)

                # Caminho para o arquivo .npy de saída
                dataset_output_path = os.path.join(output_dir, f"{root_dir_name}.npy")

                try:
                    # Parsear o arquivo de log
                    queue = parse_log_file(log_file_path)

                    # Associar nuvens de pontos a posições globais
                    dataset = associate_pointclouds_with_positions(queue)

                    # Preparar o dataset numpy
                    dataset_np = []

                    # Processar cada nuvem de pontos
                    for pc_file, position in dataset:
                        # Remover o segmento "/_lidar/" do caminho relativo, se presente
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

                        # Adicionar ao dataset numpy
                        dataset_np.append({
                            "timestamp": position[0],
                            "x": position[1],
                            "y": position[2],
                            "theta": position[3],
                            "pointcloud": points
                        })

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