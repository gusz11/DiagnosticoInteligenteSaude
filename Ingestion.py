import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import kagglehub


path = kagglehub.dataset_download("emirhanakku/synthetic-medical-triage-priority-dataset")
print("Caminho dos arquivos:", path)

arquivos = os.listdir(path)
print("Arquivos encontrados:", arquivos)

csv_file = [f for f in arquivos if f.endswith('.csv')][0]
full_path = os.path.join(path, csv_file)

df = pd.read_csv(full_path)

df.head()

df.info()

df.to_csv("dataset/triagem_fuzzy.csv", index=False)