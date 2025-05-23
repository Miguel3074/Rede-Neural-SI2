import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, utils, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import shap # type: ignore
import os # Para criar diretório de gráficos
import traceback # Para melhor debugging de exceções

# Para reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

# --- Criar diretório para salvar os gráficos ---
output_dir = "graficos_pesquisa"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Diretório '{output_dir}' criado para salvar os gráficos.")

# --- 1. Carregamento de Dados ---
print("### 1. Carregando Dados de Treinamento ###")
# Definir nomes das colunas explicitamente, pois os CSVs não têm cabeçalho
col_names_treino = ['i', 'si1', 'si2', 'qPA', 'pulso', 'respiracao', 'gravidade', 'classe']
try:
    df_train_full = pd.read_csv('treino_sinais_vitais_com_label.csv', header=None, names=col_names_treino)

except FileNotFoundError:
    print("Erro: Arquivo 'treino_sinais_vitais_com_label.csv' não encontrado.")
    exit()
except pd.errors.EmptyDataError:
    print("Erro: Arquivo 'treino_sinais_vitais_com_label.csv' está vazio.")
    exit()
except Exception as e:
    print(f"Erro ao carregar 'treino_sinais_vitais_com_label.csv': {e}")
    traceback.print_exc()
    exit()


print("Dados de treino carregados. Primeiras linhas:")
print(df_train_full.head())
print(f"Colunas do DataFrame de treino: {df_train_full.columns.tolist()}")


# Definir features e alvos com base nos nomes das colunas fornecidos
feature_names = ['qPA', 'pulso', 'respiracao']
target_regression_name = 'gravidade'
target_classification_ref_name = 'classe'
id_column_name_train = 'i' # Usando o nome definido em col_names_treino

# Verificar se as colunas esperadas existem (deve estar ok agora)
expected_cols_train = [id_column_name_train] + feature_names + [target_regression_name, target_classification_ref_name]
for col in expected_cols_train:
    if col not in df_train_full.columns:
        print(f"Erro CRÍTICO: A coluna '{col}' esperada não foi encontrada após nomeação explícita. Verifique 'col_names_treino'.")
        exit()

X_full = df_train_full[feature_names].copy()
y_grav_full = df_train_full[target_regression_name].copy()
y_class_full = df_train_full[target_classification_ref_name].copy()

# --- 2. Pré-processamento e Preparação ---
print("\n### 2. Pré-processamento ###")
from sklearn.model_selection import train_test_split # type: ignore
X_train, X_val, y_train_grav, y_val_grav = train_test_split(
    X_full, y_grav_full, test_size=0.2, random_state=42
)

print(f"Tamanho do conjunto de treino: {X_train.shape[0]}")
print(f"Tamanho do conjunto de validação: {X_val.shape[0]}")

# Normalização das Features usando a camada Normalization do Keras
normalizer = layers.Normalization(axis=-1)
normalizer.adapt(X_train.to_numpy()) # Convertendo para NumPy array

print("Médias para normalização (após adaptação):", normalizer.mean.numpy())
print("Variâncias para normalização (após adaptação):", normalizer.variance.numpy())

# --- 3. Construção do Modelo de Regressão (Keras) ---
print("\n### 3. Construção do Modelo de Regressão ###")
K.clear_session() # Limpa a sessão Keras anterior, se houver

model_regression = models.Sequential([
    normalizer, # Camada de normalização adaptada
    layers.Dense(64, activation='relu', name='Camada_Oculta_1'),
    layers.Dense(32, activation='relu', name='Camada_Oculta_2'),
    layers.Dense(1, activation='linear', name='Camada_Saida_Gravidade')
], name="Modelo_Regressao_Gravidade")

model_regression.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

try:
    if not model_regression.built:
        model_regression.build(input_shape=(None, len(feature_names)))
    model_regression.summary()

    plot_model_path = os.path.join(output_dir, "arquitetura_rede_neural.png")
    utils.plot_model(
        model_regression,
        to_file=plot_model_path,
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        rankdir='TB' 
    )
    print(f"Diagrama da rede neural salvo em: {plot_model_path}")
except Exception as e:
    print(f"Erro ao gerar o diagrama da rede (verifique se Graphviz e pydot estão instalados e no PATH): {e}")
    print("Detalhes do erro:")
    traceback.print_exc()


# --- 4. Treinamento do Modelo ---
print("\n### 4. Treinamento do Modelo ###")
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

history = model_regression.fit(
    X_train, y_train_grav,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, y_val_grav),
    callbacks=[early_stopping],
    verbose=1
)

# Plotar histórico de treinamento
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perda Treino')
plt.plot(history.history['val_loss'], label='Perda Validação')
plt.title('Perda (MSE) durante Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='MAE Treino')
plt.plot(history.history['val_mean_absolute_error'], label='MAE Validação')
plt.title('Erro Absoluto Médio (MAE) durante Treinamento')
plt.xlabel('Épocas')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plot_hist_path = os.path.join(output_dir, "historico_treinamento.png")
plt.savefig(plot_hist_path)
print(f"Gráfico do histórico de treinamento salvo em: {plot_hist_path}")
plt.show()

# --- 5. Determinação dos Limiares de Classificação ---
print("\n### 5. Determinação dos Limiares de Classificação ###")
gravidades_por_classe = {1: [], 2: [], 3: [], 4: []}
for index, row in df_train_full.iterrows():
    g = row[target_regression_name]
    c = int(row[target_classification_ref_name])
    if c in gravidades_por_classe:
        gravidades_por_classe[c].append(g)

for c_key in list(gravidades_por_classe.keys()):
    if not gravidades_por_classe[c_key]:
        print(f"Aviso: Classe {c_key} não possui dados de gravidade no conjunto de treino. Removendo para cálculo de limiares.")
        del gravidades_por_classe[c_key]

thresholds = {}
missing_data_for_threshold = False
default_grav_val = 0 if len(y_grav_full) == 0 else np.median(y_grav_full)

if 1 in gravidades_por_classe and 2 in gravidades_por_classe and gravidades_por_classe[1] and gravidades_por_classe[2]:
    thresholds['T1'] = (max(gravidades_por_classe[1]) + min(gravidades_por_classe[2])) / 2
else:
    print("Aviso: Dados insuficientes para calcular T1. Usando fallback.")
    missing_data_for_threshold = True
    thresholds['T1'] = np.percentile(y_grav_full, 25) if len(y_grav_full) > 0 else default_grav_val + 5

if 2 in gravidades_por_classe and 3 in gravidades_por_classe and gravidades_por_classe[2] and gravidades_por_classe[3]:
    thresholds['T2'] = (max(gravidades_por_classe[2]) + min(gravidades_por_classe[3])) / 2
else:
    print("Aviso: Dados insuficientes para calcular T2. Usando fallback.")
    missing_data_for_threshold = True
    thresholds['T2'] = np.percentile(y_grav_full, 50) if len(y_grav_full) > 0 else default_grav_val + 10

if 3 in gravidades_por_classe and 4 in gravidades_por_classe and gravidades_por_classe[3] and gravidades_por_classe[4]:
    thresholds['T3'] = (max(gravidades_por_classe[3]) + min(gravidades_por_classe[4])) / 2
else:
    print("Aviso: Dados insuficientes para calcular T3. Usando fallback.")
    missing_data_for_threshold = True
    thresholds['T3'] = np.percentile(y_grav_full, 75) if len(y_grav_full) > 0 else default_grav_val + 15

print(f"Limiares determinados: {thresholds}")
if missing_data_for_threshold:
    print("ALERTA: Um ou mais limiares podem ter usado fallbacks.")

def classificar_por_gravidade(gravidade_valor, t1, t2, t3):
    if gravidade_valor < t1: return 1
    elif gravidade_valor < t2: return 2
    elif gravidade_valor < t3: return 3
    else: return 4

plt.figure(figsize=(10, 6))
for classe_val_key in sorted(gravidades_por_classe.keys()):
    if gravidades_por_classe[classe_val_key]:
        plt.hist(gravidades_por_classe[classe_val_key], bins=15, alpha=0.7, label=f'Classe {classe_val_key} (Real)')

if 'T1' in thresholds: plt.axvline(thresholds['T1'], color='red', linestyle='--', label=f'T1 ({thresholds["T1"]:.2f})')
if 'T2' in thresholds: plt.axvline(thresholds['T2'], color='green', linestyle='--', label=f'T2 ({thresholds["T2"]:.2f})')
if 'T3' in thresholds: plt.axvline(thresholds['T3'], color='blue', linestyle='--', label=f'T3 ({thresholds["T3"]:.2f})')
plt.title('Distribuição da Gravidade Real por Classe e Limiares')
plt.xlabel('Gravidade Real')
plt.ylabel('Frequência')
plt.legend()
plot_dist_path = os.path.join(output_dir, "distribuicao_gravidade_classes.png")
plt.savefig(plot_dist_path)
print(f"Gráfico da distribuição de gravidade salvo em: {plot_dist_path}")
plt.show()

# --- 6. Avaliação do Modelo de Regressão ---
print("\n### 6. Avaliação do Modelo de Regressão (no conjunto de validação) ###")
loss_val, mae_val = model_regression.evaluate(X_val, y_val_grav, verbose=0)
print(f"Perda (MSE) no conjunto de validação: {loss_val:.4f}")
print(f"Erro Absoluto Médio (MAE) no conjunto de validação: {mae_val:.4f}")

y_pred_val_grav = model_regression.predict(X_val).flatten()
plt.figure(figsize=(8, 8))
plt.scatter(y_val_grav, y_pred_val_grav, alpha=0.5, label='Predições vs Reais')
min_val_plot = min(y_val_grav.min(), y_pred_val_grav.min()) if len(y_val_grav)>0 and len(y_pred_val_grav)>0 else 0
max_val_plot = max(y_val_grav.max(), y_pred_val_grav.max()) if len(y_val_grav)>0 and len(y_pred_val_grav)>0 else 100
plt.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], color='red', linestyle='--', label='Linha Ideal (y=x)')
plt.xlabel('Gravidade Real (Validação)')
plt.ylabel('Gravidade Predita (Validação)')
plt.title('Gravidade Predita vs. Real (Validação)')
plt.axis('equal'); plt.axis('square'); plt.legend()
plot_pred_real_path = os.path.join(output_dir, "predicao_vs_real_validacao.png")
plt.savefig(plot_pred_real_path)
print(f"Gráfico de predição vs. real salvo em: {plot_pred_real_path}")
plt.show()

# --- 7. Interpretabilidade com SHAP ---
print("\n### 7. Interpretabilidade com SHAP ###")
predict_fn_shap = lambda x_input: model_regression.predict(x_input)
background_sample_shap = shap.sample(X_train, min(100, X_train.shape[0]))

try:
    explainer = shap.KernelExplainer(predict_fn_shap, background_sample_shap)
    print("Calculando valores SHAP (pode levar alguns minutos)...")
    X_val_sample_shap = X_val.sample(min(100, X_val.shape[0]), random_state=42)
    shap_values = explainer.shap_values(X_val_sample_shap)

    if isinstance(shap_values, list): shap_values_data = shap_values[0]
    else: shap_values_data = shap_values

    print("Valores SHAP calculados. Gerando plots.")
    plt.figure()
    shap.summary_plot(shap_values_data, X_val_sample_shap, feature_names=feature_names, show=False)
    plt.title("Importância das Features (SHAP Summary Plot)")
    plot_shap_summary_path = os.path.join(output_dir, "shap_summary_plot.png")
    plt.savefig(plot_shap_summary_path, bbox_inches='tight')
    print(f"Gráfico SHAP Summary salvo em: {plot_shap_summary_path}")
    plt.show()

    plt.figure()
    shap.summary_plot(shap_values_data, X_val_sample_shap, feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Importância Média das Features (SHAP Bar Plot)")
    plot_shap_bar_path = os.path.join(output_dir, "shap_bar_plot.png")
    plt.savefig(plot_shap_bar_path, bbox_inches='tight')
    print(f"Gráfico SHAP Bar salvo em: {plot_shap_bar_path}")
    plt.show()
except Exception as e:
    print(f"Erro ao calcular ou plotar valores SHAP: {e}")
    traceback.print_exc()

# --- 8. Predição no Conjunto de Teste Cego ---
print("\n### 8. Predição no Conjunto de Teste Cego ###")
arquivo_teste = 'treino_sinais_vitais_sem_label.csv'
resultados_finais = []

# Definir nomes das colunas para o arquivo de teste, assumindo a mesma estrutura do treino, mas sem 'gravidade' e 'classe'
# A ordem é: i, si1, si2, qPA, pulso, respiracao (6 colunas)
# Se o seu arquivo de teste tiver um número diferente de colunas ou ordem, ajuste col_names_teste
col_names_teste = ['i', 'si1', 'si2', 'qPA', 'pulso', 'respiracao']

try:
    df_teste = pd.read_csv(arquivo_teste, header=None, names=col_names_teste)
    print(f"Dados de teste '{arquivo_teste}' carregados. Primeiras linhas:")
    print(df_teste.head())
    print(f"Colunas do DataFrame de teste: {df_teste.columns.tolist()}")

    # Verificar se as colunas de features esperadas existem
    for col in feature_names: # feature_names = ['qPA', 'pulso', 'respiracao']
        if col not in df_teste.columns:
            print(f"Erro CRÍTICO: A coluna de feature '{col}' esperada não foi encontrada no arquivo de teste após nomeação. Verifique 'col_names_teste'.")
            exit()
    
    id_column_name_test = 'i' # Usando o nome definido em col_names_teste
    if id_column_name_test not in df_teste.columns:
        print(f"Erro CRÍTICO: A coluna de ID '{id_column_name_test}' não foi encontrada no arquivo de teste. Verifique 'col_names_teste'.")
        exit()
    print(f"Usando '{id_column_name_test}' como coluna de ID para o arquivo de teste.")

    X_teste_features = df_teste[feature_names].copy()
    predicoes_gravidade_teste = model_regression.predict(X_teste_features).flatten()

    t1_final = thresholds.get('T1', np.percentile(y_grav_full, 25) if len(y_grav_full) > 0 else default_grav_val + 5)
    t2_final = thresholds.get('T2', np.percentile(y_grav_full, 50) if len(y_grav_full) > 0 else default_grav_val + 10)
    t3_final = thresholds.get('T3', np.percentile(y_grav_full, 75) if len(y_grav_full) > 0 else default_grav_val + 15)

    if missing_data_for_threshold:
        print(f"ALERTA (Teste): Usando limiares com possíveis fallbacks: T1={t1_final:.2f}, T2={t2_final:.2f}, T3={t3_final:.2f}")

    predicoes_classe_teste = [classificar_por_gravidade(g, t1_final, t2_final, t3_final) for g in predicoes_gravidade_teste]

    print("\nResultados para o arquivo de teste:")
    print("i,gravid,classe")
    for idx, (i_val, grav, classe) in enumerate(zip(df_teste[id_column_name_test], predicoes_gravidade_teste, predicoes_classe_teste)):
        linha_resultado = f"{i_val},{grav:.4f},{classe}"
        resultados_finais.append(linha_resultado)
        if idx < 10 or (idx >= len(df_teste)-5 and len(df_teste)>10):
             print(linha_resultado)

    output_pred_path = os.path.join(output_dir, "predicoes_teste_final.txt")
    with open(output_pred_path, "w") as f_out:
        f_out.write("i,gravid,classe\n")
        for linha in resultados_finais:
            f_out.write(linha + "\n")
    print(f"\nResultados da predição do teste salvos em: {output_pred_path}")

except FileNotFoundError:
    print(f"ERRO: Arquivo de teste '{arquivo_teste}' não encontrado.")
except pd.errors.EmptyDataError:
    print(f"Erro: Arquivo de teste '{arquivo_teste}' está vazio.")
    exit()
except Exception as e:
    print(f"ERRO ao processar o arquivo de teste: {e}")
    traceback.print_exc()

print("\n--- Processo Concluído ---")
print(f"Todos os gráficos e predições foram salvos no diretório: '{output_dir}'")
