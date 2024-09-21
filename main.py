import cv2
import numpy as np
import pandas as pd

# Carregar os dados do CSV
df = pd.read_csv('rangemap_1000_rows.csv', header=0)

# Exibir as primeiras linhas do DataFrame para verificar o conteúdo
print("Primeiras linhas do DataFrame:")
print(df.head())

# Converter colunas específicas para numérico e lidar com erros
df[['x', 'y', 'pos_x', 'pos_y']] = df[['x', 'y', 'pos_x', 'pos_y']].apply(pd.to_numeric, errors='coerce')

# Verificar se há pelo menos 4 pares de pontos
if len(df.dropna()) < 4:
    raise ValueError("É necessário ter pelo menos 4 pares de pontos para calcular a homografia.")

# Extrair os pontos de origem (x, y) e de destino (pos_x, pos_y)
pontos_origem = df[['x', 'y']].dropna().values.astype(np.float32)
pontos_destino = df[['pos_x', 'pos_y']].dropna().values.astype(np.float32)

# Calcular a matriz de homografia
matriz_homografia, status = cv2.findHomography(pontos_origem, pontos_destino, cv2.RANSAC)

if matriz_homografia is None:
    raise ValueError("Não foi possível calcular a matriz de homografia.")

# Verificar se todos os pontos contribuíram para a homografia
if not all(status.flatten()):
    print("Nem todos os pontos foram utilizados para calcular a homografia.")

# Novo ponto a ser transformado (Substitua pelos valores reais)
novo_X = 1059  # Exemplo de valor
novo_Y = 1366  # Exemplo de valor
novo_ponto = np.array([[novo_X, novo_Y]], dtype=np.float32)

# Aplicar a matriz de homografia
ponto_transformado = cv2.perspectiveTransform(np.array([novo_ponto]), matriz_homografia)

# Extrair as coordenadas transformadas
pos_x_transformado, pos_y_transformado = ponto_transformado[0][0]

# Mostrar o ponto transformado com o texto formatado
print(f'Ponto X transformado: {pos_x_transformado}, Ponto Y transformado: {pos_y_transformado}')