import cv2
import numpy as np
import pandas as pd

# 1. Carregar o dataset para gerar a matriz de homografia
df_train = pd.read_csv('rangemap_1000_rows.csv', header=0, low_memory=False)

# Converter colunas para numérico e lidar com NaN (coerce força a conversão de dados inválidos para NaN)
df_train[['x', 'y', 'pos_x', 'pos_y']] = df_train[['x', 'y', 'pos_x', 'pos_y']].apply(pd.to_numeric, errors='coerce')

# Verificar se há pelo menos 4 pares de pontos válidos
if len(df_train.dropna()) < 4:
    raise ValueError("É necessário ter pelo menos 4 pares de pontos para calcular a homografia.")

# Extrair pontos de origem e destino, removendo NaN
pontos_origem = df_train[['x', 'y']].dropna().values.astype(np.float32)
pontos_destino = df_train[['pos_x', 'pos_y']].dropna().values.astype(np.float32)

# Calcular a matriz de homografia com RANSAC
matriz_homografia, status = cv2.findHomography(pontos_origem, pontos_destino, cv2.RANSAC)

if matriz_homografia is None:
    raise ValueError("Não foi possível calcular a matriz de homografia.")

# 2. Carregar o dataset de teste para estimar pos_x e pos_y
df_test = pd.read_csv('rangemap_100_rows.csv', header=0, low_memory=False)

# Converter colunas para numérico e lidar com NaN
df_test[['x', 'y', 'pos_x', 'pos_y']] = df_test[['x', 'y', 'pos_x', 'pos_y']].apply(pd.to_numeric, errors='coerce')

# Inicializar lista para armazenar resultados, precisão e os erros para cada eixo
estimativas = []
precisoes_x = []
precisoes_y = []

MSE=0
counter=0

# 3. Ler linha por linha, estimar pos_x e pos_y e calcular precisão
for index, row in df_test.iterrows():
    # Pegar o ponto X e Y
    ponto_original = np.array([[row['x'], row['y']]], dtype=np.float32)
    
    # Transformar o ponto com a matriz de homografia
    ponto_estimado = cv2.perspectiveTransform(np.array([ponto_original]), matriz_homografia)
    
    # Extrair pos_x e pos_y estimados
    pos_x_est = ponto_estimado[0][0][0]
    pos_y_est = ponto_estimado[0][0][1]
    
    # Adicionar ao dataframe os valores estimados
    estimativas.append([row['x'], row['y'], row['pos_x'], row['pos_y'], pos_x_est, pos_y_est])
    
    # Calcular a precisão da estimativa (erro percentual entre valor real e estimado)
    erro_x = abs(row['pos_x'] - pos_x_est)
    erro_y = abs(row['pos_y'] - pos_y_est)
    
    # Se os valores reais forem zero, evitamos divisão por zero
    if row['pos_x'] != 0:
        precisao_x = 100 * (1 - erro_x / abs(row['pos_x']))
    else:
        precisao_x = 100
    
    if row['pos_y'] != 0:
        precisao_y = 100 * (1 - erro_y / abs(row['pos_y']))
    else:
        precisao_y = 100
    
    precisoes_x.append([row['x'], row['y'], row['pos_x'], pos_x_est, precisao_x])
    precisoes_y.append([row['x'], row['y'], row['pos_y'], pos_y_est, precisao_y])

# Criar DataFrames para as precisões de pos_x e pos_y
df_precisao_x = pd.DataFrame(precisoes_x, columns=['x', 'y', 'pos_x_real', 'pos_x_est', 'precisao_x'])
df_precisao_y = pd.DataFrame(precisoes_y, columns=['x', 'y', 'pos_y_real', 'pos_y_est', 'precisao_y'])

# 4. Salvar as tabelas separadas para pos_x e pos_y
df_precisao_x.to_csv('precisao_pos_x.csv', index=False)
df_precisao_y.to_csv('precisao_pos_y.csv', index=False)