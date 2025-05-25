import pandas as pd # type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from tensorflow.keras import models, layers, utils, optimizers # type: ignore
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt # type: ignore
import shap # type: ignore
import os 
import traceback
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import accuracy_score, classification_report # type: ignore

RANDOM_SEED = 42
OUTPUT_DIR = "graficos_pesquisa"
ARQUIVO_TREINO_LABELS = 'treino_sinais_vitais_com_label.csv'
ARQUIVO_TESTE_SEM_LABELS = 'treino_sinais_vitais_sem_label.csv'
ARQUIVO_PREDICOES_FINAL = "predicoes_teste_final.txt"
ARQUIVO_MODELO_REDE_PNG = "arquitetura_rede_neural.png"

COL_NAMES_TREINO = ['i', 'si1', 'si2', 'qPA', 'pulso', 'respiracao', 'gravidade', 'classe']
COL_NAMES_TESTE = ['i', 'si1', 'si2', 'qPA', 'pulso', 'respiracao', 'gravidade_real_no_arquivo_teste']

FEATURE_NAMES = ['qPA', 'pulso', 'respiracao']
TARGET_REGRESSION_NAME = 'gravidade' 
TARGET_CLASSIFICATION_REF_NAME = 'classe' 
ID_COLUMN_NAME = 'i' 

TEST_SPLIT_SIZE = 0.2 
NEURONS_LAYER_1 = 64
ACTIVATION_LAYER_1 = 'relu'
NEURONS_LAYER_2 = 32
ACTIVATION_LAYER_2 = 'relu'
OUTPUT_ACTIVATION_REGRESSION = 'linear'
LEARNING_RATE = 0.001
LOSS_FUNCTION_REGRESSION = 'mean_squared_error'
METRICS_REGRESSION = ['mean_absolute_error']
EPOCHS = 200
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 20 
SHAP_BACKGROUND_SAMPLES = 100
SHAP_EXPLAIN_SAMPLES = 100
HIST_BINS = 15
THRESHOLD_T1_FALLBACK_PERCENTILE = 25
THRESHOLD_T2_FALLBACK_PERCENTILE = 50
THRESHOLD_T3_FALLBACK_PERCENTILE = 75
THRESHOLD_FALLBACK_OFFSET_T1 = 5
THRESHOLD_FALLBACK_OFFSET_T2 = 10
THRESHOLD_FALLBACK_OFFSET_T3 = 15

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Diretório '{OUTPUT_DIR}' criado para salvar os gráficos.")

print("### 1. Carregando Dados de Treinamento ###")
try:
    df_train_full = pd.read_csv(ARQUIVO_TREINO_LABELS, header=None, names=COL_NAMES_TREINO)
except FileNotFoundError:
    print(f"Erro: Arquivo de treino '{ARQUIVO_TREINO_LABELS}' não encontrado.")
    exit()
except pd.errors.EmptyDataError:
    print(f"Erro: Arquivo de treino '{ARQUIVO_TREINO_LABELS}' está vazio.")
    exit()
except Exception as e:
    print(f"Erro ao carregar '{ARQUIVO_TREINO_LABELS}': {e}")
    traceback.print_exc(); exit()

print("Dados de treino carregados. Primeiras linhas:")
print(df_train_full.head())
print(f"Colunas do DataFrame de treino: {df_train_full.columns.tolist()}")

expected_cols_train = [ID_COLUMN_NAME] + FEATURE_NAMES + [TARGET_REGRESSION_NAME, TARGET_CLASSIFICATION_REF_NAME]
for col in expected_cols_train:
    if col not in df_train_full.columns:
        print(f"Erro CRÍTICO: A coluna '{col}' esperada não foi encontrada no arquivo de treino. Verifique 'COL_NAMES_TREINO'. Colunas encontradas: {df_train_full.columns.tolist()}")
        exit()

X_full = df_train_full[FEATURE_NAMES].copy()
y_grav_full = df_train_full[TARGET_REGRESSION_NAME].copy()
y_class_full = df_train_full[TARGET_CLASSIFICATION_REF_NAME].copy() 

print("\n### 2. Pré-processamento ###")
X_train, X_val, y_train_grav, y_val_grav = train_test_split(
    X_full, y_grav_full, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_SEED
)
print(f"Tamanho do conjunto de treino: {X_train.shape[0]}")
print(f"Tamanho do conjunto de validação: {X_val.shape[0]}")

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(X_train.to_numpy())
print("Médias para normalização (após adaptação):", normalizer.mean.numpy())
print("Variâncias para normalização (após adaptação):", normalizer.variance.numpy())

print("\n### 3. Construção do Modelo de Regressão ###")
K.clear_session()
model_regression = models.Sequential([
    normalizer,
    layers.Dense(NEURONS_LAYER_1, activation=ACTIVATION_LAYER_1, name='Camada_Oculta_1'),
    layers.Dense(NEURONS_LAYER_2, activation=ACTIVATION_LAYER_2, name='Camada_Oculta_2'),
    layers.Dense(1, activation=OUTPUT_ACTIVATION_REGRESSION, name='Camada_Saida_Gravidade')
], name="Modelo_Regressao_Gravidade")
model_regression.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=LOSS_FUNCTION_REGRESSION, metrics=METRICS_REGRESSION)
try:
    if not model_regression.built: model_regression.build(input_shape=(None, len(FEATURE_NAMES)))
    model_regression.summary()
    plot_model_path = os.path.join(OUTPUT_DIR, ARQUIVO_MODELO_REDE_PNG)
    utils.plot_model( model_regression, to_file=plot_model_path, show_shapes=True, show_layer_names=True, show_layer_activations=True, rankdir='TB')
    print(f"Diagrama da rede neural salvo em: {plot_model_path}")
except Exception as e:
    print(f"Erro ao gerar o diagrama da rede: {e}"); traceback.print_exc()

print("\n### 4. Treinamento do Modelo ###")
early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)
history = model_regression.fit(
    X_train, y_train_grav, epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val_grav), callbacks=[early_stopping], verbose=1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); plt.plot(history.history['loss'], label='Perda Treino'); plt.plot(history.history['val_loss'], label='Perda Validação')
plt.title('Perda (MSE) durante Treinamento'); plt.xlabel('Épocas'); plt.ylabel('Perda'); plt.legend()
plt.subplot(1, 2, 2); plt.plot(history.history['mean_absolute_error'], label='MAE Treino'); plt.plot(history.history['val_mean_absolute_error'], label='MAE Validação')
plt.title('Erro Absoluto Médio (MAE) durante Treinamento'); plt.xlabel('Épocas'); plt.ylabel('MAE'); plt.legend()
plt.tight_layout(); plot_hist_path = os.path.join(OUTPUT_DIR, "historico_treinamento.png")
plt.savefig(plot_hist_path); print(f"Gráfico do histórico de treinamento salvo em: {plot_hist_path}"); plt.show()

print("\n### 5. Determinação dos Limiares de Classificação ###")
gravidades_por_classe = {i: [] for i in range(1, 5)}
for index, row in df_train_full.iterrows():
    g = row[TARGET_REGRESSION_NAME]; c = int(row[TARGET_CLASSIFICATION_REF_NAME])
    if c in gravidades_por_classe: gravidades_por_classe[c].append(g)
for c_key in list(gravidades_por_classe.keys()):
    if not gravidades_por_classe[c_key]: print(f"Aviso: Classe {c_key} sem dados."); del gravidades_por_classe[c_key]
thresholds = {}; missing_data_for_threshold = False
default_grav_val = 0 if len(y_grav_full) == 0 else np.median(y_grav_full)

def get_threshold_val(cls1_vals, cls2_vals, fallback_percentile, fallback_offset):
    global missing_data_for_threshold
    if cls1_vals and cls2_vals: return (max(cls1_vals) + min(cls2_vals)) / 2
    else: print(f"Aviso: Dados insuficientes para limiar. Usando fallback."); missing_data_for_threshold = True
    return np.percentile(y_grav_full, fallback_percentile) if len(y_grav_full) > 0 else default_grav_val + fallback_offset

thresholds['T1'] = get_threshold_val(gravidades_por_classe.get(1), gravidades_por_classe.get(2), THRESHOLD_T1_FALLBACK_PERCENTILE, THRESHOLD_FALLBACK_OFFSET_T1)
thresholds['T2'] = get_threshold_val(gravidades_por_classe.get(2), gravidades_por_classe.get(3), THRESHOLD_T2_FALLBACK_PERCENTILE, THRESHOLD_FALLBACK_OFFSET_T2)
thresholds['T3'] = get_threshold_val(gravidades_por_classe.get(3), gravidades_por_classe.get(4), THRESHOLD_T3_FALLBACK_PERCENTILE, THRESHOLD_FALLBACK_OFFSET_T3)

t1_final_calc = thresholds['T1']; t2_final_calc = thresholds['T2']; t3_final_calc = thresholds['T3']
print(f"Limiares determinados: T1={t1_final_calc:.2f}, T2={t2_final_calc:.2f}, T3={t3_final_calc:.2f}")
if missing_data_for_threshold: print("ALERTA: Um ou mais limiares podem ter usado fallbacks.")

def classificar_por_gravidade(gravidade_valor, t1, t2, t3):
    if gravidade_valor < t1:
        return 1
    elif gravidade_valor < t2:
        return 2
    elif gravidade_valor < t3:
        return 3
    else:
        return 4

plt.figure(figsize=(10, 6))
for classe_val_key in sorted(gravidades_por_classe.keys()):
    if gravidades_por_classe[classe_val_key]: plt.hist(gravidades_por_classe[classe_val_key], bins=HIST_BINS, alpha=0.7, label=f'Classe {classe_val_key} (Real)')
plt.axvline(t1_final_calc, color='red', linestyle='--', label=f'T1 ({t1_final_calc:.2f})')
plt.axvline(t2_final_calc, color='green', linestyle='--', label=f'T2 ({t2_final_calc:.2f})')
plt.axvline(t3_final_calc, color='blue', linestyle='--', label=f'T3 ({t3_final_calc:.2f})')
plt.title('Distribuição da Gravidade Real por Classe e Limiares'); plt.xlabel('Gravidade Real'); plt.ylabel('Frequência'); plt.legend()
plot_dist_path = os.path.join(OUTPUT_DIR, "distribuicao_gravidade_classes.png")
plt.savefig(plot_dist_path); print(f"Gráfico da distribuição de gravidade salvo em: {plot_dist_path}"); plt.show()

print("\n### 6. Avaliação do Modelo de Regressão (no conjunto de validação) ###")
loss_val, mae_val = model_regression.evaluate(X_val, y_val_grav, verbose=0)
print(f"Perda (MSE) no conjunto de validação: {loss_val:.4f}"); print(f"Erro Absoluto Médio (MAE) no conjunto de validação: {mae_val:.4f}")
y_pred_val_grav = model_regression.predict(X_val).flatten()
print(f"Algumas gravidades preditas (validação): {y_pred_val_grav[:10]}")
plt.figure(figsize=(8, 8)); plt.scatter(y_val_grav, y_pred_val_grav, alpha=0.5, label='Predições vs Reais')
min_val_plot = min(y_val_grav.min(), y_pred_val_grav.min()) if len(y_val_grav)>0 and len(y_pred_val_grav)>0 else 0
max_val_plot = max(y_val_grav.max(), y_pred_val_grav.max()) if len(y_val_grav)>0 and len(y_pred_val_grav)>0 else 100
plt.plot([min_val_plot, max_val_plot], [min_val_plot, max_val_plot], color='red', linestyle='--', label='Linha Ideal (y=x)')
plt.xlabel('Gravidade Real (Validação)'); plt.ylabel('Gravidade Predita (Validação)')
plt.title('Gravidade Predita vs. Real (Validação)'); plt.axis('equal'); plt.axis('square'); plt.legend()
plot_pred_real_path = os.path.join(OUTPUT_DIR, "predicao_vs_real_validacao.png")
plt.savefig(plot_pred_real_path); print(f"Gráfico de predição vs. real salvo em: {plot_pred_real_path}"); plt.show()

print("\n### 7. Interpretabilidade com SHAP ###")
predict_fn_shap = lambda x_input: model_regression.predict(x_input)
background_sample_shap = shap.sample(X_train, min(SHAP_BACKGROUND_SAMPLES, X_train.shape[0]))
try:
    explainer = shap.KernelExplainer(predict_fn_shap, background_sample_shap)
    print("Calculando valores SHAP (pode levar alguns minutos)...")
    X_val_sample_shap = X_val.sample(min(SHAP_EXPLAIN_SAMPLES, X_val.shape[0]), random_state=RANDOM_SEED)
    shap_values = explainer.shap_values(X_val_sample_shap)
    if isinstance(shap_values, list): shap_values_data = shap_values[0]
    else: shap_values_data = shap_values
    print("Valores SHAP calculados. Gerando plots.")
    plt.figure(); shap.summary_plot(shap_values_data, X_val_sample_shap, feature_names=FEATURE_NAMES, show=False)
    plt.title("Importância das Features (SHAP Summary Plot)")
    plot_shap_summary_path = os.path.join(OUTPUT_DIR, "shap_summary_plot.png")
    plt.savefig(plot_shap_summary_path, bbox_inches='tight')
    print(f"Gráfico SHAP Summary salvo em: {plot_shap_summary_path}"); plt.show()
except Exception as e:
    print(f"Erro ao calcular ou plotar valores SHAP: {e}"); traceback.print_exc()

print("\n### 8. Predição no Conjunto de Teste Cego ###")
resultados_finais = []
try:
    df_teste = pd.read_csv(ARQUIVO_TESTE_SEM_LABELS, header=None, names=COL_NAMES_TESTE)
    print(f"Dados de teste '{ARQUIVO_TESTE_SEM_LABELS}' carregados. Primeiras linhas:")
    print(df_teste.head())
    print(f"Colunas do DataFrame de teste: {df_teste.columns.tolist()}")

    if ID_COLUMN_NAME not in df_teste.columns:
        print(f"Erro CRÍTICO: Coluna de ID '{ID_COLUMN_NAME}' não encontrada no df_teste. Verifique 'COL_NAMES_TESTE'. Colunas existentes: {df_teste.columns.tolist()}"); exit()
    for col in FEATURE_NAMES:
        if col not in df_teste.columns:
            print(f"Erro CRÍTICO: Coluna de feature '{col}' não encontrada no df_teste. Verifique 'COL_NAMES_TESTE'. Colunas existentes: {df_teste.columns.tolist()}"); exit()
    
    X_teste_features = df_teste[FEATURE_NAMES].copy()
    predicoes_gravidade_teste = model_regression.predict(X_teste_features).flatten()
    print(f"Algumas gravidades preditas (teste): {predicoes_gravidade_teste[:10]}")

    print(f"Limiares usados para teste: T1={t1_final_calc:.2f}, T2={t2_final_calc:.2f}, T3={t3_final_calc:.2f}")
    if missing_data_for_threshold: print(f"ALERTA (Teste): Usando limiares com possíveis fallbacks.")

    predicoes_classe_teste = np.array([classificar_por_gravidade(g, t1_final_calc, t2_final_calc, t3_final_calc) for g in predicoes_gravidade_teste])
    print(f"Algumas classes preditas (teste): {predicoes_classe_teste[:10]}")

    print("\nResultados para o arquivo de teste:")
    print("i,gravid,classe")
    for idx, (i_val, grav, classe_pred_val) in enumerate(zip(df_teste[ID_COLUMN_NAME], predicoes_gravidade_teste, predicoes_classe_teste)):
        linha_resultado = f"{i_val},{grav:.4f},{classe_pred_val}"
        resultados_finais.append(linha_resultado)
        if idx < 10 or (idx >= len(df_teste)-5 and len(df_teste)>10): print(linha_resultado)

    output_pred_path = os.path.join(OUTPUT_DIR, ARQUIVO_PREDICOES_FINAL)
    with open(output_pred_path, "w") as f_out:
        f_out.write("i,gravid,classe\n")
        for linha in resultados_finais: f_out.write(linha + "\n")
    print(f"\nResultados da predição do teste salvos em: {output_pred_path}")

    print("\n### 9. Avaliação da Classificação no 'Teste Cego' (usando gabarito do treino) ###")
    if len(df_teste) == len(y_class_full): 
        true_classes_for_test_set = y_class_full.to_numpy()
        
        accuracy = accuracy_score(true_classes_for_test_set, predicoes_classe_teste)
        
        labels_present = sorted(np.unique(np.concatenate((true_classes_for_test_set, predicoes_classe_teste))))
        target_names_report = [f'Classe {i}' for i in labels_present]
        
        report_str = classification_report(
            true_classes_for_test_set,
            predicoes_classe_teste,
            labels=labels_present, 
            target_names=target_names_report,
            zero_division=0 
        )
        
        print(f"Acurácia da Classificação no conjunto de teste (comparado ao gabarito do treino): {accuracy*100:.2f}%")
        print("\nRelatório de Classificação Detalhado:")
        print(report_str)
        
        report_path = os.path.join(OUTPUT_DIR, "relatorio_classificacao_teste.txt")
        with open(report_path, "w") as f_report:
            f_report.write(f"Acurácia da Classificação: {accuracy*100:.2f}%\n\n")
            f_report.write("Relatório de Classificação Detalhado:\n")
            f_report.write(report_str)
        print(f"Relatório de classificação salvo em: {report_path}")

        report_dict = classification_report(
            true_classes_for_test_set,
            predicoes_classe_teste,
            labels=labels_present,
            target_names=target_names_report,
            zero_division=0,
            output_dict=True
        )

        metrics_to_plot = ['precision', 'recall', 'f1-score']
        plot_data = {metric: [] for metric in metrics_to_plot}
        class_labels_for_plot = []

        for class_name_report in target_names_report:
            if class_name_report in report_dict:
                class_labels_for_plot.append(class_name_report)
                for metric in metrics_to_plot:
                    plot_data[metric].append(report_dict[class_name_report][metric])
        if class_labels_for_plot: 
            x_axis = np.arange(len(class_labels_for_plot))
            width = 0.25 
            multiplier = 0

            fig, ax = plt.subplots(figsize=(12, 7))

            for metric_name, metric_values in plot_data.items():
                offset = width * multiplier
                rects = ax.bar(x_axis + offset, metric_values, width, label=metric_name.capitalize())
                ax.bar_label(rects, padding=3, fmt='%.2f')
                multiplier += 1

            ax.set_ylabel('Pontuação')
            ax.set_title('Métricas de Classificação por Classe (Precision, Recall, F1-score)')
            ax.set_xticks(x_axis + width * (len(metrics_to_plot) -1) / 2 , class_labels_for_plot)
            ax.legend(loc='upper left', ncols=len(metrics_to_plot))
            ax.set_ylim(0, 1.1) 

            plt.tight_layout()
            plot_report_grafico_path = os.path.join(OUTPUT_DIR, "grafico_relatorio_classificacao.png")
            plt.savefig(plot_report_grafico_path)
            print(f"Gráfico do relatório de classificação salvo em: {plot_report_grafico_path}")
            plt.show()
        else:
            print("Não foi possível gerar o gráfico do relatório de classificação: nenhuma classe com dados no relatório.")
    else:
        print("Não foi possível calcular a acurácia da classificação no 'teste cego':")
        print(f"O número de linhas no arquivo de teste ({len(df_teste)}) não corresponde ao arquivo de treino original ({len(y_class_full)}).")
        print("Certifique-se de que os arquivos correspondem às mesmas instâncias para comparação direta.")

except FileNotFoundError: print(f"ERRO: Arquivo de teste '{ARQUIVO_TESTE_SEM_LABELS}' não encontrado.")
except pd.errors.EmptyDataError: print(f"Erro: Arquivo de teste '{ARQUIVO_TESTE_SEM_LABELS}' está vazio."); exit()
except Exception as e: print(f"ERRO ao processar o arquivo de teste: {e}"); traceback.print_exc()

print("\n--- Processo Concluído ---")
print(f"Todos os gráficos e predições foram salvos no diretório: '{OUTPUT_DIR}'")
