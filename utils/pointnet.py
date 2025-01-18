import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# 1. Seleção da Região de Interesse (ROI)
def segment_road_with_ransac(pcd, x_range=(-70, 70), y_range=(-10, 10)):
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox.min_bound = [x_range[0], y_range[0], -float("inf")]
    bbox.max_bound = [x_range[1], y_range[1], float("inf")]
    roi_cloud = pcd.crop(bbox)

    plane_model, inlier_indices = roi_cloud.segment_plane(
        distance_threshold=0.07, ransac_n=3, num_iterations=200
    )
    road_cloud = roi_cloud.select_by_index(inlier_indices)
    return road_cloud

# 2. Detecção de Pontos de Meio-Fio
def detect_curb_points(road_cloud, segment_length=0.4, density_threshold=3):
    points = np.asarray(road_cloud.points)
    y_coords = points[:, 1]
    segments = np.arange(y_coords.min(), y_coords.max(), segment_length)

    curb_points = []
    road_points = []

    for i in range(len(segments) - 1):
        segment_mask = (y_coords >= segments[i]) & (y_coords < segments[i + 1])
        segment_points = points[segment_mask]
        density = len(segment_points) / (segments[i + 1] - segments[i])

        if density > density_threshold:
            curb_points.append(segment_points)
        else:
            road_points.append(segment_points)

    curb_points = np.vstack(curb_points) if curb_points else np.empty((0, 3))
    road_points = np.vstack(road_points) if road_points else np.empty((0, 3))

    curb_cloud = o3d.geometry.PointCloud()
    curb_cloud.points = o3d.utility.Vector3dVector(curb_points)

    road_cloud.points = o3d.utility.Vector3dVector(road_points)

    return road_cloud, curb_cloud

# 3. Detecção de Marcações de Faixa
def detect_lane_markings(road_cloud):
    points = np.asarray(road_cloud.points)
    if points.size == 0:
        print("Aviso: Nenhuma nuvem de pontos para processar.")
        return o3d.geometry.PointCloud()  # Retorna uma nuvem vazia

    z_coords = points[:, 2]
    threshold = np.mean(z_coords) + 1.5 * np.std(z_coords)
    lane_markings = points[z_coords > threshold]

    if lane_markings.size == 0:
        print("Aviso: Nenhuma marcação de faixa detectada.")
        return o3d.geometry.PointCloud()  # Retorna uma nuvem de pontos vazia

    lane_cloud = o3d.geometry.PointCloud()
    lane_cloud.points = o3d.utility.Vector3dVector(lane_markings)
    return lane_cloud

# 4. Dataset para PyTorch
class LaneDataset(Dataset):
    def __init__(self, directory, labels):
        self.files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.ply')]
        self.labels = labels  # Lista de rótulos correspondentes aos arquivos
        self.data = []

        for file in self.files:
            pcd = o3d.io.read_point_cloud(file)
            road_cloud = segment_road_with_ransac(pcd)
            road_cloud, _ = detect_curb_points(road_cloud)
            lane_cloud = detect_lane_markings(road_cloud)
            self.data.append(np.asarray(lane_cloud.points))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        points = self.data[idx]
        label = self.labels[idx]
        return torch.tensor(points, dtype=torch.float32).T, torch.tensor(label, dtype=torch.float32)

# 5. PointNet para Regressão
class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.max(x, 2)[0]  # Pooling máximo
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 6. Treinamento Supervisionado
def train_pointnet(directory, labels, num_epochs=10, batch_size=4, learning_rate=0.001):
    dataset = LaneDataset(directory, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = PointNet()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for points, label in dataloader:
            optimizer.zero_grad()

            if points.size(2) == 0:
                print("Aviso: Batch vazio ignorado.")
                continue

            points = points.transpose(1, 2)  # Formato [Batch, 3, N]
            outputs = model(points).squeeze()  # Saída do modelo
            loss = loss_fn(outputs, label)  # Comparar saída com rótulo
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

    torch.save(model.state_dict(), "pointnet_model.pth")
    print("Modelo salvo como pointnet_model.pth")

# 7. Predição Supervisionada e Visualização
def predict_and_visualize(model_path, ply_file):
    model = PointNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    pcd = o3d.io.read_point_cloud(ply_file)
    road_cloud = segment_road_with_ransac(pcd)
    road_cloud, _ = detect_curb_points(road_cloud)
    lane_cloud = detect_lane_markings(road_cloud)

    points = np.asarray(lane_cloud.points)
    if points.size == 0:
        print("Aviso: Nenhuma marcação de faixa detectada na nuvem de pontos.")
        return

    points_tensor = torch.tensor(points, dtype=torch.float32).T.unsqueeze(0)  # Formato [1, 3, N]

    if points_tensor.size(2) == 0:
        print("Aviso: Nenhum ponto válido para predição.")
        return

    with torch.no_grad():
        prediction = model(points_tensor).item()
    print(f"Predição do modelo: {prediction}")

    o3d.visualization.draw_geometries([lane_cloud])


def main():
    script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_path, "../data/pointclouds")
    labels = []
    print(data_path, labels)
    train_pointnet(data_path, num_epochs=1)

    # Exemplo de uso para predição e visualização
    ply_file = data_path + "/1682191700.ply"
    predict_and_visualize("pointnet_model.pth", ply_file)

if __name__ == "__main__":
    main()