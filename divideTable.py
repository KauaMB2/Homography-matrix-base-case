import pandas as pd

# Função para realizar a leitura, criação do CSV e remoção das linhas lidas do arquivo original
def process_dataset(filename, num_rows, output_filename):
    # Carregar o dataset original
    df = pd.read_csv(filename, header=None, low_memory=False)

    # Definir o nome das colunas
    df.columns = ['x', 'y', 'range', 'pos_x', 'pos_y', 'lat', 'lon']

    # Verificar se há linhas suficientes no dataset original
    if len(df) < num_rows:
        print(f"Não há linhas suficientes no arquivo para selecionar {num_rows} linhas.")
        return

    # Selecionar linhas aleatórias do dataset original
    df_sample = df.sample(n=num_rows, random_state=42)

    # Salvar as linhas selecionadas em um novo arquivo CSV com o cabeçalho
    df_sample.to_csv(output_filename, index=False, header=True)

    # Remover as linhas selecionadas do dataset original
    df_remaining = df.drop(df_sample.index)

    # Salvar o dataset restante de volta no arquivo original
    df_remaining.to_csv(filename, index=False, header=False)

    print(f"{num_rows} linhas salvas em {output_filename} e removidas de {filename}.")

# Arquivo original
filename = 'rangemap_full__.csv'

# Processar as diferentes quantidades de linhas
process_dataset(filename, 1663900, 'rangemap_1663900.csv')
