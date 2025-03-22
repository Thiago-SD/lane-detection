import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split

# Função para calcular a distância de um ponto a um segmento de reta
def distance_to_segment(point, segment_start, segment_end):
    """
    Calcula a distância de um ponto a um segmento de reta.
    :param point: Ponto (x, y).
    :param segment_start: Ponto inicial do segmento (x, y).
    :param segment_end: Ponto final do segmento (x, y).
    :return: Distância do ponto ao segmento.
    """
    def dot(v, w):
        return v[0] * w[0] + v[1] * w[1]

    def length(v):
        return np.sqrt(v[0] ** 2 + v[1] ** 2)

    def vector(p1, p2):
        return [p2[0] - p1[0], p2[1] - p1[1]]

    def point_to_line_distance(p, l1, l2):
        line_vec = vector(l1, l2)
        p_vec = vector(l1, p)
        line_len = length(line_vec)
        if line_len == 0:
            return length(p_vec)
        t = max(0, min(1, dot(p_vec, line_vec) / (line_len ** 2)))
        projection = [l1[0] + t * line_vec[0], l1[1] + t * line_vec[1]]
        return length(vector(p, projection))

    return point_to_line_distance(point, segment_start, segment_end)


# Função para preparar os dados
def prepare_data(datasets, base_index):
    """
    Prepara os dados para treinamento.
    :param datasets: Lista de datasets carregados.
    :param base_index: Índice da instância base.
    :return: X (nuvens de pontos), y (distâncias).
    """
    X = []
    y = []

    # Escolher a instância base
    base_instance = datasets[base_index][0]  # Primeira entrada do dataset base
    base_pointcloud = base_instance['pointcloud']
    base_position = (base_instance['x'], base_instance['y'])  # Posição global do veículo

    # Calcular as distâncias para todas as instâncias
    for dataset in datasets:
        for entry in dataset:
            pointcloud = entry['pointcloud']
            position = (entry['x'], entry['y'])  # Posição global do veículo

            # Calcular a distância ao segmento de reta da instância base
            distance = distance_to_segment(position, base_position, (base_position[0] + 1, base_position[1] + 1))
            X.append(pointcloud)
            y.append(distance)

    return np.array(X), np.array(y)


# Carregar os datasets .npy
def load_datasets(data_dir):
    """
    Carrega todos os datasets .npy de um diretório.
    :param data_dir: Diretório contendo os arquivos .npy.
    :return: Lista de datasets.
    """
    datasets = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".npy"):
            dataset = np.load(os.path.join(data_dir, file_name), allow_pickle=True)
            datasets.append(dataset)
    return datasets


# Definir o modelo PointNet (usando a implementação anterior)
class PointNet(Model):
    def __init__(self, num_points, name="PointNet"):
        super(PointNet, self).__init__(name=name)
        self.num_points = num_points

        # Camadas do PointNet
        self.conv1 = layers.Conv1D(64, 1, activation='relu')
        self.conv2 = layers.Conv1D(128, 1, activation='relu')
        self.conv3 = layers.Conv1D(1024, 1, activation='relu')
        self.max_pool = layers.GlobalMaxPooling1D()
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.fc3 = layers.Dense(1, activation=None)  # Saída para regressão

        # Batch normalization
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.bn3 = layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.max_pool(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# Treinamento do modelo
def train_model(X, y, num_points, epochs=50, batch_size=32):
    """
    Treina o modelo PointNet.
    :param X: Nuvens de pontos.
    :param y: Distâncias.
    :param num_points: Número de pontos em cada nuvem.
    :param epochs: Número de épocas.
    :param batch_size: Tamanho do batch.
    """
    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Construir o modelo
    model = PointNet(num_points=num_points)
    model.build(input_shape=(None, num_points, 3))
    model.compile(optimizer='adam', loss='mse')

    # Treinar o modelo
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    return model, history


# Função principal
if __name__ == "__main__":
    # Diretório contendo os datasets .npy
    data_dir = "data/output"

    # Carregar os datasets
    datasets = load_datasets(data_dir)

    # Preparar os dados (escolher a primeira instância como base)
    X, y = prepare_data(datasets, base_index=0)

    # Número de pontos em cada nuvem de pontos
    num_points = X.shape[1]

    # Treinar o modelo
    model, history = train_model(X, y, num_points=num_points, epochs=50, batch_size=32)

    # Salvar o modelo treinado
    model.save("pointnet_regression_model.h5")