.PHONY: all install system-deps test clean run help

# Configurações
VENV = pointnet_env
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
SCRIPT_DIR = utils

# Comando principal - executa tudo
all: system-deps test dataset preprocess train

# Cria o venv se não existir e instala dependências Python
activate: 
	$(VENV)/bin/activate

install:
	@echo "Criando virtual environment..."
	python3 -m venv $(VENV)
	@echo "Instalando dependências Python..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo " Virtual environment criado e dependências Python instaladas"

# Instala dependências do sistema (Graphviz)
system-deps: install
	@echo "Instalando Graphviz no sistema..."
	sudo apt update && sudo apt install -y graphviz
	@echo " Graphviz instalado"

# Testa se todas as dependências estão funcionando
test: activate
	@echo "Verificando instalações..."
	$(PYTHON) -c "import graphviz; print(' Graphviz:', graphviz.version())"
	$(PYTHON) -c "import torch; print(' PyTorch:', torch.__version__)"
	$(PYTHON) -c "import torchviz; print(' Torchviz OK')"
	$(PYTHON) -c "import sklearn; print(' Scikit-learn OK')"
	$(PYTHON) -c "import numpy; print(' Numpy OK')"
	$(PYTHON) -c "import matplotlib; print(' Matplotlib OK')"
	@echo "Todas as dependências verificadas!"

# Executa o pré-processamento
preprocess: activate
	@echo "Executando pré-processamento..."
	$(PYTHON) $(SCRIPT_DIR)/preprocessor.py

# Executa o treinamento
train: activate
	@echo "Executando treinamento..."
	$(PYTHON) $(SCRIPT_DIR)/pointnet.py

# Executa o dataset (se necessário)
dataset: activate
	@echo "Executando processamento do dataset..."
	$(PYTHON) $(SCRIPT_DIR)/dataset.py

# Limpeza completa
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete
	@echo " Ambiente limpo"

# Ajuda
help:
	@echo "Comandos disponíveis:"
	@echo "  make all        - Setup completo (venv + dependências + testes)"
	@echo "  make install    - Apenas dependências Python"
	@echo "  make system-deps - Apenas Graphviz no sistema"
	@echo "  make test       - Testa todas as dependências"
	@echo "  make preprocess - Executa pré-processamento"
	@echo "  make train      - Executa treinamento"
	@echo "  make dataset    - Executa processamento do dataset"
	@echo "  make clean      - Remove venv e arquivos temporários"
	@echo "  make activate   - Comando para ativar o venv"
	@echo "  make help       - Mostra esta ajuda"
