import os
import numpy as np
import pandas as pd
import csv
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

# Função para ler o arquivo .pointcloud
def readLidarFile(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
        return data
    except Exception as e:
        print(f"Erro ao ler o arquivo {file_path}: {e}")
        return None

# Função para converter dados binários em pontos 3D
def binaryTo3d(file_path, n_rays):
    # Verificar se n_rays é um número inteiro válido
    if not isinstance(n_rays, int) or n_rays <= 0:
        raise ValueError(f"n_rays deve ser um número inteiro positivo. Recebido: {n_rays}")

    # Tamanho dos tipos de dados
    size_double = np.dtype(np.float64).itemsize  # 8 bytes
    size_short = np.dtype(np.int16).itemsize  # 2 bytes
    size_char = np.dtype(np.uint8).itemsize  # 1 byte

    # Tamanho de cada registro (disparo)
    record_size = size_double + (32 * size_short) + (32 * size_char)

    # Verificar se o tamanho do arquivo é consistente com n_rays
    file_size = os.path.getsize(file_path)
    expected_size = n_rays * record_size
    if file_size < expected_size:
        raise ValueError(f"Arquivo corrompido ou incompleto: tamanho esperado {expected_size}, tamanho real {file_size}")

    # Inicializar array para armazenar os pontos 3D
    points = np.empty((n_rays * 32, 3), dtype=np.float32)  # (N, 3) onde N = n_rays * 32

    # Ler o arquivo binário e processar os dados em blocos
    with open(file_path, 'rb') as f:
        for i in range(n_rays):
            # Ler o ângulo (double, 8 bytes)
            angle = np.frombuffer(f.read(size_double), dtype=np.float64)[0]

            # Ler as distâncias (32 * short, 2 bytes cada)
            distances = np.frombuffer(f.read(32 * size_short), dtype=np.int16)

            # Ler as intensidades (32 * char, 1 byte cada)
            intensities = np.frombuffer(f.read(32 * size_char), dtype=np.uint8)

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
    return points

# Função para converter graus e minutos para graus decimais
def convert_degmin_to_decimal(degmin):
    degrees = int(degmin // 100)
    minutes = degmin - (degrees * 100)
    return degrees + (minutes / 60)

# Função para converter coordenadas geodésicas (lat, lon) para UTM
def geodetic_to_utm(lat, lon):
    k0 = 0.9996
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    e = math.sqrt(1 - (b / a) ** 2)
    n = (a - b) / (a + b)
    nu = a / math.sqrt(1 - (e * math.sin(lat_rad)) ** 2)
    p = lon_rad - math.floor((lon_rad + math.pi) / (2 * math.pi)) * (2 * math.pi)

    A = a * (1 - n + (5 / 4) * (n ** 2 - n ** 3) + (81 / 64) * (n ** 4 - n ** 5))
    B = (3 / 2) * a * (n - n ** 2 + (7 / 8) * (n ** 3 - n ** 4) + (55 / 64) * n ** 5)
    C = (15 / 16) * a * (n ** 2 - n ** 3 + (3 / 4) * (n ** 4 - n ** 5))
    D = (35 / 48) * a * (n ** 3 - n ** 4 + (11 / 16) * n ** 5)
    E = (315 / 512) * a * (n ** 4 - n ** 5)

    S = A * lat_rad - B * math.sin(2 * lat_rad) + C * math.sin(4 * lat_rad) - D * math.sin(6 * lat_rad) + E * math.sin(8 * lat_rad)
    K1 = S * k0
    K2 = nu * math.sin(lat_rad) * math.cos(lat_rad) * k0 / 2
    K3 = (nu * math.sin(lat_rad) * math.cos(lat_rad) ** 3 * k0 / 24) * (5 - math.tan(lat_rad) ** 2 + 9 * e ** 2 * math.cos(lat_rad) ** 2 + 4 * e ** 4 * math.cos(lat_rad) ** 4)
    K4 = nu * math.cos(lat_rad) * k0
    K5 = (nu * math.cos(lat_rad) ** 3 * k0 / 6) * (1 - math.tan(lat_rad) ** 2 + e ** 2 * math.cos(lat_rad) ** 2)

    x = K1 + K2 * p ** 2 + K3 * p ** 4
    y = K4 * p + K5 * p ** 3 + 500000

    return x, y

# Função para ler a mensagem NMEAGGA e converter para UTM
def read_gps(line, gps_to_use=1):
    parts = line.strip().split()
    if len(parts) < 15 or parts[0] != "NMEAGGA":
        return None

    gps_id = int(parts[1])
    if gps_id != gps_to_use:
        return None

    lat_dm = float(parts[3])
    lat_orientation = parts[4]
    lon_dm = float(parts[5])
    lon_orientation = parts[6]
    altitude = float(parts[11])
    logger_timestamp = float(parts[-1])

    # Converter latitude e longitude para graus decimais
    lat = convert_degmin_to_decimal(lat_dm)
    lon = convert_degmin_to_decimal(lon_dm)

    if lat_orientation == 'S':
        lat = -lat
    if lon_orientation == 'W':
        lon = -lon

    # Converter para UTM
    x, y = geodetic_to_utm(lat, lon)

    return x, y, altitude, logger_timestamp

# Função para parsear o arquivo de log
def parse_log_file(log_file_path):
    if not os.path.exists(log_file_path):
        raise FileNotFoundError(f"Arquivo de log não encontrado: {log_file_path}")

    pointcloud_files = []
    global_positions = []

    with open(log_file_path, 'r') as f:
        for line in f:
            if line.startswith("VELODYNE_PARTIAL_SCAN_IN_FILE"):
                parts = line.strip().split()
                relative_path = parts[1]
                n_rays = int(parts[2])  # Extrair n_rays da linha do log
                logger_timestamp = float(parts[-1])
                pointcloud_files.append((logger_timestamp, relative_path, n_rays))

            elif line.startswith("NMEAGGA"):
                gps_data = read_gps(line)
                if gps_data is not None:
                    x, y, z, timestamp = gps_data
                    global_positions.append((timestamp, x, y, z))

    return pointcloud_files, global_positions

# Função para associar nuvens de pontos a posições globais
def associate_pointclouds_with_positions(pointcloud_files, global_positions):
    dataset = []

    for pc_timestamp, pc_file, n_rays in pointcloud_files:
        # Encontrar a posição global mais próxima no tempo
        closest_position = min(global_positions, key=lambda x: abs(x[0] - pc_timestamp))
        dataset.append((pc_file, n_rays, closest_position))

    return dataset

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
                output_dir = os.path.join(root, "output")
                # Verificar se o diretório de saída existe
                os.makedirs(output_dir, exist_ok=True)

                # Caminho para o arquivo .csv de saída
                dataset_output_path = os.path.join(output_dir, f"{root_dir_name}.csv")

                try:
                    # Parsear o arquivo de log
                    pointcloud_files, global_positions = parse_log_file(log_file_path)

                    # Associar nuvens de pontos a posições globais
                    dataset = associate_pointclouds_with_positions(pointcloud_files, global_positions)

                    # Abrir o arquivo .csv para escrita
                    with open(dataset_output_path, 'w', newline='') as csvfile:
                        # Definir as colunas do arquivo .csv
                        fieldnames = ["timestamp", "latitude", "longitude", "altitude", "pointcloud"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        # Escrever o cabeçalho do arquivo .csv
                        writer.writeheader()

                        # Processar cada nuvem de pontos
                        for pc_file, n_rays, position in dataset:
                            pc_timestamp, latitude, longitude, altitude = position

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
                            points = binaryTo3d(pc_path, n_rays)

                            # Escrever os dados no arquivo .csv
                            writer.writerow({
                                "timestamp": pc_timestamp,
                                "latitude": latitude,
                                "longitude": longitude,
                                "altitude": altitude,
                                "pointcloud": points.tolist()  # Converter array numpy para lista
                            })

                    print(f"Dataset salvo em {dataset_output_path}")

                except FileNotFoundError as e:
                    print(f"Erro: {e}")
                except Exception as e:
                    print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()