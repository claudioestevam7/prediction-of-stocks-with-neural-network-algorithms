# Use a imagem base Python
FROM python:3.11.2

# Defina o diretório de trabalho dentro do contêiner
WORKDIR /modelagem

# Copie o arquivo requirements.txt para o contêiner
COPY requirements.txt .

# Instale as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie o código para o contêiner
COPY . .

# Comando para executar o código Python
CMD ["python", "Modelagem.py"]