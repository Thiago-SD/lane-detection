import os
import open3d as o3d
import numpy as np

class Point:
    def __init__(self, x, y, z, r, g, b):
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.g = g
        self.b = b
    def __str__(self):
        return f"{self.x} {self.y} {self.z} {self.r} {self.g} {self.b}"
    def getXYZ(self):
        return (self.x, self.y, self.z)

velodyne_vertical_angles = [
		-30.6700000, -29.3300000, -28.0000000, -26.6700000, -25.3300000, -24.0000000, -22.6700000, -21.3300000,
		-20.0000000, -18.6700000, -17.3300000, -16.0000000, -14.6700000, -13.3300000, -12.0000000, -10.6700000,
		-9.3299999, -8.0000000, -6.6700001, -5.3299999, -4.0000000, -2.6700001, -1.3300000, 0.0000000, 1.3300000,
		2.6700001, 4.0000000, 5.3299999, 6.6700001, 8.0000000, 9.3299999, 10.6700000
]


velodyne_ray_order = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]



def readLidarFile(file_path):
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        #print(data)
    except Exception as e:
        print(f"Erro ao ler o arquivo {file_path}: {e}")
        data = None
    return data

def readLidarFileAndWriteToPLY(file_path, ply_file_path):
    try:
        data = np.fromfile(file_path, dtype=np.uint8)
        points = binaryTo3d(data)  # Converter dados binários em pontos 3D
        xyz = np.array([point.getXYZ() for point in points])  # Extrair coordenadas XYZ

        # Criar nuvem de pontos e salvar no formato .ply
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(xyz)
        o3d.io.write_point_cloud(ply_file_path, point_cloud, write_ascii=True)
    except Exception as e:
        print(f"Erro ao processar o arquivo {file_path}: {e}")

def readLidarFolder(base_path, ply_output_dir, extension=".pointcloud"):
    os.makedirs(ply_output_dir, exist_ok=True)  # Garantir que o diretório de saída exista

    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                subdir_name = os.path.basename(root)  # Nome do subdiretório atual
                ply_file_path = os.path.join(ply_output_dir, f"{subdir_name}.ply")

                # Processar e gerar o arquivo .ply
                readLidarFileAndWriteToPLY(file_path, ply_file_path) 

def computePointVelodyne (v_angle, h_angle, radius, intensity):
    cos_rot_angle = np.cos(h_angle)
    sin_rot_angle = np.sin(h_angle)

    cos_vert_angle = np.cos(v_angle)
    sin_vert_angle = np.sin(v_angle)

    xy_distance = radius * cos_vert_angle

    return Point((xy_distance * cos_rot_angle), (xy_distance * sin_rot_angle), (radius * sin_vert_angle), intensity, intensity, intensity)

def binaryTo3d (binary_data):
    size_double = np.dtype(np.float64).itemsize  # 8 bytes
    size_unsigned_short = np.dtype(np.uint16).itemsize  # 2 bytes
    size_unsigned_char = np.dtype(np.uint8).itemsize  # 1 byte

    record_size = size_double + (32 * size_unsigned_short) + (32 * size_unsigned_char)

    n_rays = len(binary_data) // record_size

    points = []

    for i in range(n_rays):
        offset = i * record_size
        h_angle = np.frombuffer(binary_data, dtype=np.float64, count=1, offset=offset)[0]
        offset += size_double
        
        distances = np.frombuffer(binary_data, dtype=np.uint16, count=32, offset=offset)
        offset += 32 * size_unsigned_short
        
        intensities = np.frombuffer(binary_data, dtype=np.uint8, count=32, offset=offset)
        
        h_angle = np.pi * h_angle / 180.
        
        for j in range(32):
            l_range = distances[velodyne_ray_order[j]] / 500.
            v_angle = velodyne_vertical_angles[j]
            v_angle = np.pi * v_angle / 180.
            points.append (computePointVelodyne(v_angle, -h_angle, l_range, intensities[velodyne_ray_order[j]]))

    return points

# Função para escrever o cabeçalho do arquivo .ply
def write_ply_header(file_path, num_points):
    with open(file_path, "w") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {num_points}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")

# Função para adicionar pontos dinamicamente ao arquivo .ply
def append_points_to_ply(file_path, points):
    with open(file_path, "a") as ply_file:
        for point in points:
            ply_file.write(f"{point}\n")


def visualizePointCloud(points):
    # Crie uma nuvem de pontos Open3D
    pcd = o3d.geometry.PointCloud()
    
    # Extraia coordenadas e cores dos objetos Point
    point_coords = np.array([[p.x, p.y, p.z] for p in points])
    point_colors = np.array([[p.r, p.g, p.b] for p in points])
    
    # Atribua os pontos e cores à nuvem
    pcd.points = o3d.utility.Vector3dVector(point_coords)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd])

def savePointCloudToPLY(points, output_path):
    """
    Salva os pontos 3D em um arquivo .ply.
    
    :param points: Lista de objetos Point contendo x, y, z, r, g, b.
    :param output_path: Caminho para salvar o arquivo .ply.
    """
    # Crie uma nuvem de pontos Open3D
    pcd = o3d.geometry.PointCloud()
    
    # Extraia coordenadas e cores dos objetos Point
    point_coords = np.array([[p.x, p.y, p.z] for p in points])
    point_colors = np.array([[p.r, p.g, p.b] for p in points])
    
    # Atribua os pontos e cores à nuvem
    pcd.points = o3d.utility.Vector3dVector(point_coords)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    # Salve a nuvem no formato .ply
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Arquivo salvo em: {output_path}")

def main():
    script_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_path, "../data")
    #print(data_path)

    #readLidarFolder(data_path, data_path+"/pointclouds")
    point_cloud = o3d.io.read_point_cloud(data_path+"/pointclouds/1682191900.ply")
    o3d.visualization.draw_geometries([point_cloud])

if __name__ == "__main__":
    main()