# lane-detection

## 1 - Configuração de Ambiente

Para realizar a criação de um ambiente virtual e instalação das dependências necessárias para a execução do módulo, favor seguir os passos após clonar o repositório
1. Crie um ambiente virtual com o comando: `python3 -m venv pointnet_env`
2. Ative o ambiente virtual:
    No Linux/macOS: `source pointnet_env/bin/activate`
    No Windows: `pointnet_env\Scripts\activate`
3. Instale as dependências com o comando: `pip install -r requirements.txt`

## 2 - Coleta e pré processamento de dados

Após a configuração do ambiente, será necessário preparar os dados que serão utilizados no treino do modelo

1. Baixar e descompatar dados de log da IARA (como por exemplo, os disponíveis em https://drive.google.com/drive/folders/1zEuzg3mQIVOG6q_5n-x_Fz0rClFfqzqJ) em um diretório /lane-detection/data/input
2. Depositar os arquivos contendo a captura das posições globais correspondentes ao arquivo de log descompactado no mesmo diretório, com a nomenclatura globalpos_log_volta_da_ufes_<data>.txt
3. Após estes passos, a estrutura de arquivos deverá ficar como:

![image](https://github.com/user-attachments/assets/3a00aeb1-468c-40c7-a866-f6710fdaf47c)

4. Com os arquivos nos diretórios esperados, executar o script dataset.py com `python3 dataset.py`
5. Após a execução do script, as globalpos vão ser associadas às pointclouds e salvas no subdiretório data/output, como na imagem em anexo:
   
![image](https://github.com/user-attachments/assets/d557eea4-bed6-4553-9574-b6b9e4d6100f)

6. Após isso, executar o script preprocessor.py com `python3 preprocessor.py`
7. Quando a execução for finalizada, os dados de treino se encontrarão no dataset entitulado complete_training_data.npz no diretório training_data, como em anexo:

![image](https://github.com/user-attachments/assets/79730d22-d18b-443b-b71e-4947e4f26b00)

## 3 - Treino e teste do modelo via Deep Learning

1. Executar o script pointnet.py com `python3 pointnet.py`
2. O desempenho do treino pode ser acompanhado na imagem training_metrics.png presente no mesmo diretório que o script, como no exemplo:

![image](https://github.com/user-attachments/assets/b00a7991-9ccc-4f34-a6d0-383b5da6f13c)

![image](https://github.com/user-attachments/assets/bef311d9-7452-4574-9b26-927f97ab3e81)

3. O número de épocas de execução do treino do modelo pode ser alterado via constante NUM_EPOCHS dentro do script pointnet.py:

![image](https://github.com/user-attachments/assets/c5202096-5ec2-43e7-9b08-3d733a5c7dc7)


4. Após o treino, o modelo é salvo no formato lane_distance_regressor.pth para uso futuro no mesmo diretório que o script:

   ![image](https://github.com/user-attachments/assets/8aa7a0cc-9e95-45cb-ad84-a14b47f424a5)

5. Esse modelo pode então ser usado para prever a distância até o centro da lane via dados de Lidar

