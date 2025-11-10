# lane-detection

## 1 - Configuração de Ambiente

Para realizar a criação de um ambiente virtual e instalação das dependências necessárias para a execução do módulo, favor seguir os passos após clonar o repositório. O projeto conta com um 
arquivo Makefile para deixar a configuração de ambiente conveniente, os comandos são listados a seguir:

1.  make all        - Setup completo (venv + dependências + testes)
2.  make install    - Apenas dependências Python
3.  make system-deps - Apenas Graphviz no sistema
4.  make test       - Testa todas as dependências
5.  make preprocess - Executa pré-processamento
6.  make train      - Executa treinamento
7.  make dataset    - Executa processamento do dataset
8.  make clean      - Remove venv e arquivos temporários
9.  make activate   - Comando para ativar o venv
10.  make help       - Mostra esta ajuda

Para a configuração inicial do ambiente, basta utilizar o comando

`make system-deps`

## 2 - Coleta e pré processamento de dados

Após a configuração do ambiente, será necessário preparar os dados que serão utilizados no treino do modelo

1. Baixar e descompatar dados de log da IARA (como por exemplo, os disponíveis em https://drive.google.com/drive/folders/1zEuzg3mQIVOG6q_5n-x_Fz0rClFfqzqJ) em um diretório /lane-detection/data/input
2. Depositar os arquivos contendo a captura das posições globais correspondentes ao arquivo de log descompactado no mesmo diretório, com a nomenclatura globalpos_log_volta_da_ufes_<data>.txt
3. Após estes passos, a estrutura de arquivos deverá ficar como:

![image](https://github.com/user-attachments/assets/3a00aeb1-468c-40c7-a866-f6710fdaf47c)

4. Com os arquivos nos diretórios esperados, executar o script dataset.py com `make dataset`
5. Após a execução do script, as globalpos vão ser associadas às pointclouds e salvas no subdiretório data/output, como na imagem em anexo:
   
![image](https://github.com/user-attachments/assets/d557eea4-bed6-4553-9574-b6b9e4d6100f)

6. Após isso, executar o script preprocessor.py com `make preprocess.py`
7. Quando a execução for finalizada, os dados de treino se encontrarão no dataset entitulado complete_training_data.npz no diretório training_data, como em anexo:

![image](https://github.com/user-attachments/assets/79730d22-d18b-443b-b71e-4947e4f26b00)

## 3 - Treino e teste do modelo via Deep Learning

1. Executar o script pointnet.py com `make train`
2. O desempenho do treino pode ser acompanhado na imagem training_metrics.png presente no mesmo diretório que o script, como no exemplo:

![image](https://github.com/user-attachments/assets/b00a7991-9ccc-4f34-a6d0-383b5da6f13c)

![image](https://github.com/user-attachments/assets/178b3346-62d6-47e0-ad80-aff8752f18d1)

3. O número de épocas de execução do treino do modelo pode ser alterado via constante NUM_EPOCHS dentro do script pointnet.py:

![image](https://github.com/user-attachments/assets/c5202096-5ec2-43e7-9b08-3d733a5c7dc7)


4. Após o treino, o modelo é salvo no formato lane_distance_regressor.pth para uso futuro no diretório data/models:

   ![image](https://github.com/user-attachments/assets/bac667c8-52a6-4a6a-a0ad-29f1d8377cad)

5. Esse modelo pode então ser usado para prever a distância até o centro da lane via dados de Lidar

