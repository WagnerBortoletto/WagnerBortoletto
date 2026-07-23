# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:59:24 2026
Author: Wagner Bortoletto
"""
#%% 
# ==========================================
# 1. IMPORTAÇÃO E PREPARAÇÃO
# ==========================================

import sys
print(sys.executable)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.graphics.gofplots import qqplot
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense



pd.set_option("display.max.columns", None)

#%%

#%% 
# ==========================================
# 1. CARREGAR DADOS E DESCRITIVOS
# ==========================================

# 1. Importa a tabela completa com todas as colunas
caminho_arquivo = "C:/Users/wagne/OneDrive/_Backup_/__USUAL__/_Artigos e Projetos/Antonio/Artigo CO2 Brazil/co2_fossil_plus_land_use/co2_fossil_plus_land_use_BR_1950_2024.csv"
df_completo = pd.read_csv(caminho_arquivo, sep=";", decimal=",")

print(df_completo)

df_completo = df_completo.rename(columns={'Total_Milhoes':'Total (mil)',
                                          'LandUse_Milhoes':'Land Use (mil)',
                                          'FossilFuels_Milhoes': 'Fossil Fuels (mil)'}); print(df_completo)

df_completo.head(n=5)
df_completo.tail(n=5)
df_completo.info()

# Exibe as colunas disponíveis para você conferir os nomes exatos
print("Colunas disponíveis no DataFrame:", df_completo.columns.tolist())


# Estatísticas Descritivas
numericas = df_completo[['Total (mil)', 'Land Use (mil)', 'Fossil Fuels (mil)']]
numericas = pd.DataFrame(numericas)
resumo_completo = numericas.describe(include='all')
print(resumo_completo)


## Histograma e BoxPlot ##
# Configurando a área do gráfico para ter 1 linha e 2 colunas

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Criando o Histograma (Gráfico da esquerda)
sns.histplot(data=numericas, x="Total (mil)", kde=True, ax=axes[0], color="lightgray")
axes[0].set_title("Histogram")
axes[0].set_xlabel("Total CO2 Emissions (mil)")
axes[0].set_ylabel("Frequency")

# Criando o Boxplot (Gráfico da direita)
sns.boxplot(data=numericas, y="Total (mil)", ax=axes[1], color="lightgray")
axes[1].set_title("Boxplot")
axes[1].set_ylabel("Total CO2 Emissions (mil)")

# 5. Ajustando o layout e exibindo
plt.tight_layout()
plt.show()

#%%

# ==========================================
# 3. Extraindo as séries temporais desejadas
# ==========================================

# Opção A: Extrair apenas UMA variável (Isso cria um objeto 'Series' do pandas)
# serie_unica = df_completo['Total_Milhoes']

# Opção B: Extrair MÚLTIPLAS variáveis ao mesmo tempo (Isso cria um novo 'DataFrame')
# Note que usamos colchetes duplos [[ ]] para selecionar mais de uma coluna
series_multiplas = df_completo[['Total (mil)', 'Land Use (mil)', 'Fossil Fuels (mil)']]

# Conferindo os resultados
#print("\nSérie Única:")
#print(serie_unica.head())

print("\nMúltiplas Séries:")
print(series_multiplas.head())

#%%
# ==========================================
# 4. Convertendo em Séries
# ==========================================


# 1. Criar o índice temporal (1950 até 2024)
# O parâmetro freq='YS' significa "Year Start" (1º de janeiro de cada ano)
indice_temporal = pd.date_range(start='1950', periods=75, freq='YS')

# 2. Atribuir esse índice ao seu DataFrame
# Substitua 'df_completo' pelo nome da variável onde estão as suas séries
series_multiplas.index = indice_temporal

#%%
# ==========================================
# 5. Plotar o gráfico de linhas
# ==========================================

# O comando .plot() do pandas já reconhece o índice temporal e plota todas as colunas juntas
# figsize=(12, 6) define o tamanho da imagem (largura, altura)
ax = series_multiplas.plot(figsize=(12, 6), linewidth=2)

# Configurações visuais do gráfico
plt.title('Time series of CO2 emissions in Brazil (1950 - 2024)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Values CO2 Emissions', fontsize=12)

# Adiciona uma grade no fundo para facilitar a leitura dos valores
plt.grid(True, linestyle='--', alpha=0.7)

# Organiza a legenda (útil se você tiver várias séries/colunas no DataFrame)
plt.legend(title='Variables', bbox_to_anchor=(1.05, 1), loc='upper left')

# Ajusta o layout para a legenda não ficar cortada e exibe o gráfico
plt.tight_layout()
plt.show()

#%%
# ==========================================
# 6. Decomposição
# ==========================================


# A decomposição é feita em uma variável por vez.
# Substitua 'sua_variavel' pelo nome da coluna que deseja analisar
serie_alvo = series_multiplas['Total (mil)']

# Realiza a decomposição
# 'additive' assume que os componentes se somam. Se a variação cresce com o tempo, use 'multiplicative'
# period=5 define que o ciclo sazonal ocorre a cada 4 anos
decomposicao = seasonal_decompose(serie_alvo, model='additive', period=4)


# Plota os 4 gráficos (Original, Tendência, Sazonalidade e Resíduos)
fig = decomposicao.plot()
fig.set_size_inches(12, 8) # Ajusta o tamanho para ficar legível
plt.tight_layout()
plt.show()

# Se quiser extrair os dados matemáticos separadamente para um novo DataFrame:
    
dados_decomp = pd.DataFrame({
    'Tendencia': decomposicao.trend,
    'Sazonalidade': decomposicao.seasonal,
    'Residuos': decomposicao.resid
})


resumo_decomp= dados_decomp.describe(include='all')
print(resumo_decomp)


#%%
# ==========================================
# 7. ARIMA
# ==========================================
warnings.filterwarnings("ignore")

# Teste de Dickey Fuller

resultado = adfuller(serie_alvo)
print(f"Estatística ADF: {resultado[0]:.3f}")
print(f"Valor-p: {resultado[1]:.3f}")


def testar_estacionariedade(serie_alvo):
    print("--- Teste Dickey-Fuller Aumentado (ADF) ---")
    # H0: A série possui raiz unitária (NÃO é estacionária)
    adf_result = adfuller(serie_alvo, autolag='AIC')
    print(f'Estatística ADF: {adf_result[0]:.4f}')
    print(f'P-valor: {adf_result[1]:.4f}')
    if adf_result[1] <= 0.05:
        print("Conclusão: Rejeitamos H0. A série é ESTACIONÁRIA.\n")
    else:
        print("Conclusão: Falhamos em rejeitar H0. A série NÃO É ESTACIONÁRIA.\n")

    print("--- Teste KPSS ---")
    # H0: A série é estacionária ao redor de uma tendência determinística
    kpss_result = kpss(serie_alvo, regression='c', nlags="auto")
    print(f'Estatística KPSS: {kpss_result[0]:.4f}')
    print(f'P-valor: {kpss_result[1]:.4f}')
    if kpss_result[1] <= 0.05:
        print("Conclusão: Rejeitamos H0. A série NÃO É ESTACIONÁRIA.\n")
    else:
        print("Conclusão: Falhamos em rejeitar H0. A série é ESTACIONÁRIA.\n")
        
        
        
# Executando os testes na série original
print("TESTES NA SÉRIE ORIGINAL:")
testar_estacionariedade(serie_alvo)


# Diferenciando a Série

serie_diff = serie_alvo.diff().dropna()

plt.figure(figsize=(10,4))
plt.plot(serie_diff, color="orange")
plt.title("Série Após Diferenciação (1ª Ordem)")
plt.xlabel("Tempo")
plt.ylabel("Diferença")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Novo teste ADF

def testar_estacionariedade(serie_diff):
    print("--- Teste Dickey-Fuller Aumentado (ADF) ---")
    # H0: A série possui raiz unitária (NÃO é estacionária)
    adf_result = adfuller(serie_diff, autolag='AIC')
    print(f'Estatística ADF: {adf_result[0]:.4f}')
    print(f'P-valor: {adf_result[1]:.4f}')
    if adf_result[1] <= 0.05:
        print("Conclusão: Rejeitamos H0. A série é ESTACIONÁRIA.\n")
    else:
        print("Conclusão: Falhamos em rejeitar H0. A série NÃO É ESTACIONÁRIA.\n")

    print("--- Teste KPSS ---")
    # H0: A série é estacionária ao redor de uma tendência determinística
    kpss_result = kpss(serie_diff, regression='c', nlags="auto")
    print(f'Estatística KPSS: {kpss_result[0]:.4f}')
    print(f'P-valor: {kpss_result[1]:.4f}')
    if kpss_result[1] <= 0.05:
        print("Conclusão: Rejeitamos H0. A série NÃO É ESTACIONÁRIA.\n")
    else:
        print("Conclusão: Falhamos em rejeitar H0. A série é ESTACIONÁRIA.\n")
        
        
        
# Executando os testes na série original
print("TESTES NA SÉRIE ORIGINAL:")
testar_estacionariedade(serie_diff)



# PLOTS DE AUTOCORRELAÇÃO (ACF e PACF)

plt.figure(figsize=(12, 5))

# Plotando a série original
plt.subplot(1, 2, 1)
plt.plot(serie_alvo)
plt.title('Série Temporal Original')

# Plotando a ACF-PACF da série diferenciada (ajuda a definir o termo 'p' - AR)

fig, axes = plt.subplots(1, 2, figsize=(12,4))
plot_acf(serie_diff, ax=axes[0], lags=30)
plot_pacf(serie_diff, ax=axes[1], lags=30)
axes[0].set_title("ACF - Autocorrelação")
axes[1].set_title("PACF - Autocorrelação Parcial")
plt.show()


# Definindo os parâmetros (p, d, q)
# p: lags do modelo autoregressivo (olhe a PACF)
# d: grau de diferenciação (quantas vezes fizemos df.diff())
# q: lags do erro da média móvel (olha-se a ACF)
ordem_arima = (1, 1, 0) 

print(f"Treinando modelo ARIMA com ordem {ordem_arima}...")
modelo_Arima = ARIMA(serie_alvo, order=ordem_arima)
modelo_Arima_ajustado = modelo_Arima.fit()

# Resumo do modelo (verifique os valores de P>|z| para a significância dos coeficientes)
print(modelo_Arima_ajustado.summary())


#%%
# ==========================================
# 8. SARIMAX
# ==========================================

#--------------------------------------------------------------------------------------------------------
# SÉRIE ASCENDENTE DESDE 1950 E PICO EM 2003. DEPOIS ELA DECAI ATÉ 2011 E AÍ COMEÇA A OSCILAR.
# O MODELO ARIMA NÃO ESTÁ MUITO INTERESSANTE. POR ISSO, VOU TENTAR A ABORDAGEM SARIMAX.
#--------------------------------------------------------------------------------------------------------

### O modelo ARIMA não está muito interessante. Por isso, vou tentar a abordagem SARIMAX.

Break = df_completo['Break']

if isinstance(Break, pd.DataFrame):
    Break = Break.iloc[:, 0]
else:
    Break = pd.Series(Break)

# 2. Verifica se ambas têm o mesmo tamanho (devem ter!)
if len(serie_alvo) != len(Break):
    print(f"ATENÇÃO: Tamanhos diferentes! Série tem {len(serie_alvo)} linhas e Break tem {len(Break)}.")
    # Se isso acontecer, você precisará corrigir a criação da dummy para ter o mesmo número de observações.

# 3. A MÁGICA ACONTECE AQUI: Força o índice da dummy a ser idêntico ao da série alvo
Break.index = serie_alvo.index

# Opcional: converte para float (o statsmodels lida melhor com números float do que inteiros)
Break = Break.astype(float)



# DEFINIÇÃO DOS PARÂMETROS
# ==========================================
# Ordem ARIMA (p, d, q) -> Baseada nas suas análises de ACF e PACF
ordem_arima = (1, 1, 0)

# Ordem Sazonal (P, D, Q, s) -> s é a periodicidade (ex: 12 para mensal)
ordem_sazonal = (1, 0, 0, 4) 


# TREINAMENTO DO MODELO

# 1. Garante que a variável Break seja uma Série do Pandas
# Se a sua 'Break' for um dataframe com 1 coluna, extraia a coluna.
# Se já for uma lista ou array, isso a converte para o formato certo.



# ==========================================
print("Ajustando o modelo SARIMAX com a dummy de quebra estrutural...")

# A variável 'Break' entra no parâmetro 'exog'
modelo_sarimax = SARIMAX(
    endog=serie_alvo,
    exog=Break,
    order=ordem_arima,
    seasonal_order=ordem_sazonal,
    enforce_stationarity=False,
    enforce_invertibility=False
)

resultado_sarimax = modelo_sarimax.fit(disp=False)

# Verifique o P>|z| da variável 'Break' para confirmar se ela é estatisticamente significante
print(resultado_sarimax.summary())



# Analisando os resíduos do modelo
# -------------------------------------------------------------------------
residuos = resultado_sarimax.resid

fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].plot(residuos, color="gray")
axes[0].set_title("Resíduos do Modelo")
plot_pacf(residuos, ax=axes[1], lags=30)
axes[1].set_title("PACF dos Resíduos")
plt.show()

# Não há correação serial nos resíduos. Logo o modelo está bem ajustado


# -------------------------------------------------------------------------
# TESTES DE NORMALIDADE NOS RESÍDUOS
# -------------------------------------------------------------------------

alpha = 0.05

print(f"--- ANÁLISE DE NORMALIDADE (Amostra Pequena: n = {len(residuos)}) ---")


# TESTE: Shapiro-Wilk (O mais recomendado para n = 75)
# -------------------------------------------------------------------------
sw_stat, sw_p_value = stats.shapiro(residuos)

print(f"\n[1] Teste de Shapiro-Wilk (Mais indicado)")
print(f"Estatística W: {sw_stat:.4f}")
print(f"Valor-p (p-value): {sw_p_value:.4f}")

if sw_p_value > alpha:
    print("Resultado: Não rejeitamos H0. Os resíduos seguem distribuição normal.")
else:
    print("Resultado: Rejeitamos H0. Os resíduos NÃO seguem distribuição normal.")


# TESTE: Anderson-Darling (Excelente complemento para avaliar as caudas)
# -------------------------------------------------------------------------
ad_resultado = stats.anderson(residuos, dist='norm')

print(f"\n[2] Teste de Anderson-Darling")
print(f"Estatística do teste: {ad_resultado.statistic:.4f}")

# No Anderson-Darling, comparamos a estatística com os valores críticos tabelados
# Procuramos o índice correspondente ao nível de significância de 5% (geralmente o índice 2)
indice_5pct = np.where(ad_resultado.significance_level == 5.0)[0][0]
valor_critico_5pct = ad_resultado.critical_values[indice_5pct]

print(f"Valor crítico (ao nível de 5%): {valor_critico_5pct:.4f}")

if ad_resultado.statistic < valor_critico_5pct:
    print("Resultado: Não rejeitamos H0. Os resíduos seguem distribuição normal.")
else:
    print("Resultado: Rejeitamos H0. Os resíduos NÃO seguem distribuição normal.")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

# PREVISÃO com o SARIMAX (FORECAST)
# ==========================================
passos_futuros = 6  # Quantidade de períodos que você deseja prever

# CRÍTICO: Construção da dummy futura.
# Se a quebra for um novo patamar definitivo, os próximos valores são 1.
dummy_futura = pd.DataFrame({'Break': [1] * passos_futuros})

# Se os seus dados originais possuem um DatetimeIndex, é boa prática 
# definir o índice da dummy_futura para os próximos meses/dias.
# Caso contrário, o statsmodels pode gerar um aviso de alinhamento de índices.
# Exemplo genérico de continuidade de índice:
dummy_futura.index = pd.date_range(
    start=serie_alvo.index[-1] + pd.Timedelta(days=1), # Ajuste o Timedelta se for mensal (ex: pd.DateOffset(months=1))
    periods=passos_futuros, 
    freq=serie_alvo.index.inferred_freq
)

# Gerando a previsão passando a dummy futura
previsao_Sarima = resultado_sarimax.get_forecast(steps=passos_futuros, exog=dummy_futura)
previsao_media_Sarima = previsao_Sarima .predicted_mean
intervalo_confianca = previsao_Sarima .conf_int(alpha=0.05) # Intervalo de 95%



# PLOTAGEM DOS RESULTADOS
# ==========================================
plt.figure(figsize=(12, 6))

# Extraindo os valores ajustados (previsões in-sample do treinamento)
# Usamos [1:] muitas vezes para pular o primeiro valor que costuma ser 0 no SARIMAX
valores_ajustados_sarimax = resultado_sarimax.fittedvalues

# 1. Plot da série original
plt.plot(serie_alvo.index, serie_alvo, label='Total (mil)', color='steelblue', alpha=0.6, linewidth=2)

# 2. Plot do ajuste do modelo nos dados de treinamento
plt.plot(valores_ajustados_sarimax.index, valores_ajustados_sarimax, label='Fitted Values', color='orange', linestyle='--')

# 3. Plot da previsão futura
plt.plot(previsao_media_Sarima.index, previsao_media_Sarima, label='Forecasting', color='#d62728', linewidth=2)

# 4. Marcação visual da quebra estrutural
try:
    # Identifica a primeira data onde a dummy virou 1
    data_quebra = Break[Break == 1].index[0]
    plt.axvline(x=data_quebra, color='gray', linestyle='-.', linewidth=1.0, label='Break')
except Exception as e:
    print("Aviso: Não foi possível plotar a linha de quebra.", e)

plt.title('SARIMAX: Training and Forecasting')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Values CO2 Emissions', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()


#%%
# ==========================================
# 9. TRANSFORMAÇÕES ÚTEIS
# ==========================================


## Transformando em Log

serie_log = np.log(serie_alvo)

plt.figure(figsize=(10,4))
plt.plot(serie_log, color="green")
plt.title("Série Após Transformação Logarítmica")
plt.xlabel("Tempo")
plt.ylabel("log(Valor)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


## Normalizando

scaler = MinMaxScaler()
serie_norm = pd.Series(scaler.fit_transform(serie_alvo.values.reshape(-1,1)).flatten(), index=serie_alvo.index)

plt.figure(figsize=(10,4))
plt.plot(serie_norm, color="purple")
plt.title("Série Normalizada (Escala 0–1)")
plt.xlabel("Tempo")
plt.ylabel("Valor Normalizado")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


#%%
# ==========================================
# 10. PREVISÃO COM RANDOM WALK
# ==========================================

# Previsão de Random Walk (valor anterior como previsão)
previsao_rw = serie_alvo.shift(1)

plt.figure(figsize=(10,4))
plt.plot(serie_alvo, label="Série Real", color="steelblue")
plt.plot(previsao_rw, label="Previsão Random Walk", color="orange", linestyle="--")
plt.title("Modelo Random Walk - Previsão de um Passo à Frente")
plt.xlabel('Year', fontsize=12)
plt.ylabel('Values CO2 Emissions', fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()



# PARÂMETROS DA PREVISÃO
# ==========================================
passos_futuros = 6

# O Random Walk assume que todo o futuro será igual ao último valor conhecido
ultimo_valor = serie_alvo.iloc[-1]
ultima_data = serie_alvo.index[-1]


# GERAÇÃO DO ÍNDICE FUTURO
# ==========================================
# O código tenta descobrir o padrão das datas (diário, mensal, etc.).
# Se falhar, ele usa a distância de tempo entre os dois últimos pontos.
try:
    frequencia = pd.infer_freq(serie_alvo.index)
    # Cria os passos + 1 e descarta o primeiro (que é o próprio dia atual)
    datas_futuras = pd.date_range(start=ultima_data, periods=passos_futuros + 1, freq=frequencia)[1:]
except:
    delta_tempo = ultima_data - serie_alvo.index[-2]
    datas_futuras = [ultima_data + (delta_tempo * i) for i in range(1, passos_futuros + 1)]


# CRIAÇÃO DAS SÉRIES DE MODELAGEM
# ==========================================
# A previsão futura: uma repetição do último valor
previsao_rw = pd.Series([ultimo_valor] * passos_futuros, index=datas_futuras)

# O ajuste nos dados de treino (o shift que você já estava usando)
ajuste_rw = serie_alvo.shift(1)



# PLOTAGEM DOS RESULTADOS
# ==========================================
plt.figure(figsize=(12, 6))

# 1. Plot da série original
plt.plot(serie_alvo.index, serie_alvo, label='Total (mil)', color='steelblue', alpha=0.6, linewidth=2)

# 2. Plot do ajuste do modelo nos dados de treinamento
plt.plot(ajuste_rw.index, ajuste_rw, label='Fitted Values', color='orange', linestyle='--')

# 3. Plot da previsão futura
plt.plot(previsao_rw.index, previsao_rw, label='Forecasting', color='#d62728', linewidth=2)

# Conexão visual entre o último dado real e o início da previsão (para não deixar "buraco" no gráfico)
plt.plot([ultima_data, previsao_rw.index[0]], [ultimo_valor, previsao_rw.iloc[0]], color='#d62728', linewidth=2)

# 4. Marcação visual da quebra estrutural
try:
    # Identifica a primeira data onde a dummy virou 1
    data_quebra = Break[Break == 1].index[0]
    plt.axvline(x=data_quebra, color='gray', linestyle='-.', linewidth=1.0, label='Break')
except Exception as e:
    print("Aviso: Não foi possível plotar a linha de quebra.", e)

plt.title('Random Walk: Training and Forecasting')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Values CO2 Emissions', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

#%%
# ==========================================
# 11. PREVISÃO COM HOLT WINTERS
# ==========================================

# Ajuste do modelo

modelo_hw = ExponentialSmoothing(serie_alvo, trend="add", seasonal="add", seasonal_periods=4)
ajuste_hw = modelo_hw.fit()



# PREVISÕES DO MODELO HOLT-WINTERS
# ==========================================
passos_futuros = 6  # Mantendo os 6 tempos à frente

# Extraindo os valores ajustados (previsões in-sample do treinamento)
valores_ajustados_hw = ajuste_hw.fittedvalues

# Gerando as previsões futuras (out-of-sample)
previsao_hw = ajuste_hw.forecast(steps=passos_futuros)

# ==========================================
# PLOTAGEM DOS RESULTADOS
# ==========================================
plt.figure(figsize=(12, 6))

# 1. Plot da série original
plt.plot(serie_alvo.index, serie_alvo, label='Total (mil)', color='steelblue', alpha=0.6, linewidth=2)

# 2. Plot do ajuste do modelo nos dados de treinamento
plt.plot(valores_ajustados_hw.index, valores_ajustados_hw, label='Fitted Values', color='orange', linestyle='--')

# 3. Plot da previsão futura
plt.plot(previsao_hw.index, previsao_hw, label='Forecasting', color='#d62728', linewidth=2)

# Conexão visual entre o último dado real e o início da previsão (para evitar o "buraco" no gráfico)
ultima_data = serie_alvo.index[-1]
ultimo_valor = serie_alvo.iloc[-1]
plt.plot([ultima_data, previsao_hw.index[0]], [ultimo_valor, previsao_hw.iloc[0]], color='#d62728', linewidth=2)

# 4. Marcação visual da quebra estrutural
try:
    # Identifica a primeira data onde a dummy virou 1
    data_quebra = Break[Break == 1].index[0]
    plt.axvline(x=data_quebra, color='gray', linestyle='-.', linewidth=1.0, label='Break')
except Exception as e:
    print("Aviso: Não foi possível plotar a linha de quebra.", e)

plt.title('Holt-Winters: Training and Forecasting')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Values CO2 Emissions', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()


#%%
# ======================================================
# 11. PREVISÃO COM LSTM (Rede Neural Recorrente)
# ======================================================


# Normalização
scaler = MinMaxScaler()
serie_norm = scaler.fit_transform(serie_alvo.values.reshape(-1,1))



# Criação das janelas (10 valores passados -> 1 previsão)
X, y = [], []
janela = 4
for i in range(janela, len(serie_norm)):
    X.append(serie_norm[i-janela:i])
    y.append(serie_norm[i])
X, y = np.array(X), np.array(y)

# Modelo LSTM
modelo_LSTM = Sequential([
    LSTM(450, activation="tanh", input_shape=(X.shape[1], 1)),
    Dense(1)
])
modelo_LSTM.compile(optimizer="adam", loss="mse")
modelo_LSTM.fit(X, y, epochs=150, verbose=0)



# AJUSTE DOS VALORES DE TREINO (in-sample fit)
# =====================================================
window_size = 4

# Previsão do modelo sobre os próprios dados de treino (X)
fitted_scaled = modelo_LSTM.predict(X, verbose=0)
fitted_inv = scaler.inverse_transform(fitted_scaled)

# Os primeiros 'window_size' pontos da série não têm fitted value (servem só de entrada),
# por isso o índice começa em serie_alvo.index[window_size:] -> 75 - 4 = 71 valores
valores_ajustados_LSTM = pd.Series(
    fitted_inv.flatten(),
    index=serie_alvo.index[window_size:]
)


# PREVISÃO FUTURA - 6 PASSOS À FRENTE (LSTM)
# =====================================================
n_steps_forecast = 6
ultima_janela = X[-1].flatten().tolist()

previsoes_futuras = []
for _ in range(n_steps_forecast):
    entrada = np.array(ultima_janela[-window_size:]).reshape(1, window_size, 1)
    pred = modelo_LSTM.predict(entrada, verbose=0)[0, 0]
    previsoes_futuras.append(pred)
    ultima_janela.append(pred)

previsoes_futuras = np.array(previsoes_futuras).reshape(-1, 1)
previsoes_futuras_inv = scaler.inverse_transform(previsoes_futuras)

# Gera as datas futuras a partir da última data da série real
freq = pd.infer_freq(serie_alvo.index)
datas_futuras = pd.date_range(
    start=serie_alvo.index[-1],
    periods=n_steps_forecast + 1,
    freq=freq
)[1:]  # remove a 1ª data, que já é o último ponto real

previsao_lstm = pd.Series(previsoes_futuras_inv.flatten(), index=datas_futuras)


# PLOTAGEM DOS RESULTADOS
# =====================================================
plt.figure(figsize=(12, 6))
# 1. Plot da série original
plt.plot(serie_alvo.index, serie_alvo, label='Total (mil)', color='steelblue', alpha=0.6, linewidth=2)
# 2. Plot do ajuste do modelo nos dados de treinamento
plt.plot(valores_ajustados_LSTM.index, valores_ajustados_LSTM, label='Fitted Values', color='orange', linestyle='--')
# 3. Plot da previsão futura
plt.plot(previsao_lstm.index, previsao_lstm, label='Forecasting', color='#d62728', linewidth=2)
# Conexão visual entre o último dado real e o início da previsão (para evitar o "buraco" no gráfico)
ultima_data = serie_alvo.index[-1]
ultimo_valor = serie_alvo.iloc[-1]
plt.plot([ultima_data, previsao_lstm.index[0]], [ultimo_valor, previsao_lstm.iloc[0]], color='#d62728', linewidth=2)
# 4. Marcação visual da quebra estrutural
try:
    data_quebra = Break[Break == 1].index[0]
    plt.axvline(x=data_quebra, color='gray', linestyle='-.', linewidth=1.0, label='Break')
except Exception as e:
    print("Aviso: Não foi possível plotar a linha de quebra.", e)
plt.title('LSTM: Training and Forecasting')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Values CO2 Emissions', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()


#%%
# =========================================================================
# 12. Modelo Estrutural Básico (BSM / Basic Structural Model)
# =========================================================================
"""
via Espaço de Estados e Filtro de Kalman

Componentes: nível local + tendência + sazonalidade estocástica + ruído de observação
Usa statsmodels.tsa.statespace.structural.UnobservedComponents,
que é a implementação padrão de referência (Harvey, 1989) em Python.
"""


# Carregar sua série (substitua pelo seu df já indexado por data)
# ---------------------------------------------------------------
# df deve ter uma coluna com a série e index do tipo DatetimeIndex
# Exemplo:
# df = pd.read_csv("sua_serie.csv", index_col=0, parse_dates=True)
# y = df["nome_da_coluna"]

y = serie_alvo  # <-- ajuste o nome da coluna

# Se a periodicidade não estiver definida no índice, defina explicitamente
# (ex.: mensal = "MS", trimestral = "QS", etc.)
if y.index.freq is None:
    y = y.asfreq(pd.infer_freq(y.index))



# Especificar o Modelo Estrutural Básico
# ---------------------------------------------------------------
# level='local linear trend' -> nível + inclinação estocásticos (tendência)
# seasonal -> componente sazonal estocástica (defina o período correto)
# irregular=True -> erro de observação (ruído branco)
#
# Ajuste "seasonal" ao período da sua série:
#   mensal -> 12 | trimestral -> 4 | semanal -> 52 etc.
# Se a série não tiver sazonalidade, remova o argumento seasonal.

periodo_sazonal = 4  # <-- ajuste conforme sua frequência

modelo_MEB = UnobservedComponents(
    y,
    level="local linear trend",
    seasonal=periodo_sazonal,
    irregular=True,
    stochastic_level=True,
    stochastic_trend=True,
    stochastic_seasonal=True,
)


# Estimar via máxima verossimilhança (filtro de Kalman)
# ---------------------------------------------------------------
resultado_MEB = modelo_MEB.fit(disp=False, maxiter=200)
previsao_MEB = resultado_MEB.get_forecast(steps=passos_futuros)
media_prev_MEB = previsao_MEB.predicted_mean
valores_ajustados_meb = resultado_MEB.fittedvalues

print(resultado_MEB.summary())


# Diagnósticos dos resíduos padronizados
# ---------------------------------------------------------------
resultado_MEB.plot_diagnostics(figsize=(11, 8))
plt.tight_layout()
plt.show()


# Componentes não observados extraídos (nível, tendência, sazonalidade)
# ---------------------------------------------------------------------------------
resultado_MEB.plot_components(figsize=(11, 9), legend_loc="lower right")
plt.tight_layout()
plt.show()


# Previsão fora da amostra (ex.: 6 passos à frente)
# ---------------------------------------------------------------
n_passos = 6
previsao_MEB = resultado_MEB.get_forecast(steps=n_passos)
media_prev_MEB = previsao_MEB.predicted_mean
ic_prev = previsao_MEB.conf_int(alpha=0.05)



# Ponto de quebra estrutural a destacar no gráfico
# ---------------------------------------------------------------
# Opção A: você já tem uma série dummy de quebra (1 no período de quebra, 0 no resto),
#          igual ao exemplo do LSTM (variável "Break").
# Opção B: você já sabe a data da quebra -> defina diretamente.
#
# Ajuste UMA das duas linhas abaixo conforme seu caso:

# Break = df["Break"]              # <-- Opção A: descomente se tiver a dummy
data_quebra_manual = data_quebra         # <-- Opção B: ex. pd.Timestamp("2015-01-01")


# Plot: observado x ajustado x previsão, com destaque da quebra
# ---------------------------------------------------------------
plt.figure(figsize=(12, 6))

# Série original (observada)
plt.plot(y.index, y, label="Total (mil)", color="steelblue", alpha=0.6, linewidth=2)

# Valores ajustados pelo modelo no período de treino
plt.plot(
    valores_ajustados_meb.index,
    valores_ajustados_meb,
    label="Fitted Values",
    color="orange",
    linestyle="--",
)

# Previsão futura
plt.plot(media_prev_MEB.index, media_prev_MEB, label="Forecasting", color="#d62728", linewidth=2)

# Conexão visual entre o último dado real e o início da previsão (evita "buraco" no gráfico)
ultima_data = y.index[-1]
ultimo_valor = y.iloc[-1]
plt.plot(
    [ultima_data, media_prev_MEB.index[0]],
    [ultimo_valor, media_prev_MEB.iloc[0]],
    color="#d62728",
    linewidth=2,
)

# Marcação visual da quebra estrutural
try:
    if data_quebra_manual is not None:
        data_quebra = data_quebra_manual
    else:
        data_quebra = Break[Break == 1].index[0]
    plt.axvline(x=data_quebra, color="gray", linestyle="-.", linewidth=1.0, label="Break")
except Exception as e:
    print("Aviso: Não foi possível plotar a linha de quebra.", e)


# BSM = Basic Structural Model
plt.title("BSM (Kalman Filter): Training and Forecasting")
plt.xlabel("Year", fontsize=12)
plt.ylabel("Values CO2 Emissions", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle=":", alpha=0.7)
plt.tight_layout()
plt.show()


# (Opcional) Extrair estados filtrados/suavizados manualmente
# ---------------------------------------------------------------
estados_suavizados = resultado_MEB.states.smoothed  # DataFrame com nível, tendência, sazonais
# estados_suavizados.to_csv("estados_meb.csv")


#%%
# =======================================================================================
# 13. Support Vector Machine para Séries Temporais (SVR - Support Vector Regression)
# =======================================================================================

"""
Como SVM não é um modelo de espaço de estados, a série é transformada em um
problema de aprendizado supervisionado via defasagens (lags): cada observação
y_t é prevista a partir de y_{t-1}, y_{t-2}, ..., y_{t-p}.

Usa sklearn.svm.SVR, a implementação padrão de referência em Python.
"""


# Carregar sua série (substitua pelo seu df já indexado por data)
# ---------------------------------------------------------------
# df deve ter uma coluna com a série e index do tipo DatetimeIndex
# Exemplo:
# df = pd.read_csv("sua_serie.csv", index_col=0, parse_dates=True)
# y = df["nome_da_coluna"]

y = serie_alvo.copy()  # <-- ajuste o nome da coluna

if y.index.freq is None:
    y = y.asfreq(pd.infer_freq(y.index))


# Construção das defasagens (features) para o SVR
# ---------------------------------------------------------------
n_lags = 4  # <-- ajuste conforme a memória temporal desejada (ex.: 12 se mensal com sazonalidade anual)

def criar_matriz_lags(serie: pd.Series, n_lags: int) -> pd.DataFrame:
    df_lags = pd.DataFrame({"y": serie})
    for lag in range(1, n_lags + 1):
        df_lags[f"lag_{lag}"] = serie.shift(lag)
    return df_lags.dropna()

dados_lags = criar_matriz_lags(y, n_lags)

X = dados_lags.drop(columns="y")
y_alvo = dados_lags["y"]


# Divisão treino / teste (respeitando a ordem temporal)
# ---------------------------------------------------------------
n_teste = 12  # quantidade de observações mais recentes reservadas para teste
X_treino, X_teste = X.iloc[:-n_teste], X.iloc[-n_teste:]
y_treino, y_teste = y_alvo.iloc[:-n_teste], y_alvo.iloc[-n_teste:]


# Padronização (SVR é sensível à escala das variáveis)
# ---------------------------------------------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_treino_s = scaler_X.fit_transform(X_treino)
y_treino_s = scaler_y.fit_transform(y_treino.values.reshape(-1, 1)).ravel()

X_teste_s = scaler_X.transform(X_teste)


# Ajuste de hiperparâmetros via validação cruzada temporal
# ---------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)

grade_parametros = {
    "kernel": ["rbf"],
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.01, 0.1, 1],
    "epsilon": [0.01, 0.05, 0.1],
}

busca = GridSearchCV(
    SVR(),
    grade_parametros,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)
busca.fit(X_treino_s, y_treino_s)

modelo_SVR = busca.best_estimator_
print("Melhores hiperparâmetros:", busca.best_params_)



# Avaliação no conjunto de teste
# ---------------------------------------------------------------
pred_teste_s = modelo_SVR .predict(X_teste_s)
pred_teste = scaler_y.inverse_transform(pred_teste_s.reshape(-1, 1)).ravel()

rmse = np.sqrt(np.mean((y_teste.values - pred_teste) ** 2))
mae = np.mean(np.abs(y_teste.values - pred_teste))
mape = np.mean(np.abs((y_teste.values - pred_teste) / y_teste.values)) * 100

print(f"\nDesempenho fora da amostra (teste):")
print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}%")


# Valores ajustados dentro da amostra de treino
# ---------------------------------------------------------------
pred_treino_s = modelo_SVR .predict(X_treino_s)
pred_treino = scaler_y.inverse_transform(pred_treino_s.reshape(-1, 1)).ravel()
valores_ajustados_SVR = pd.Series(pred_treino, index=y_treino.index)



# Previsão recursiva fora da amostra (h passos à frente)
# ---------------------------------------------------------------
n_passos = 6
historico = list(y.iloc[-n_lags:].values)  # últimas observações reais conhecidas
previsoes_futuras = []

for _ in range(n_passos):
    entrada = np.array(historico[-n_lags:][::-1]).reshape(1, -1)  # ordem lag_1..lag_n
    entrada_s = scaler_X.transform(entrada)
    pred_s = modelo_SVR .predict(entrada_s)
    pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()[0]
    previsoes_futuras.append(pred)
    historico.append(pred)

datas_futuras = pd.date_range(
    start=y.index[-1] + (y.index[1] - y.index[0]),
    periods=n_passos,
    freq=y.index.freq,
)
media_prev_SVR = pd.Series(previsoes_futuras, index=datas_futuras)

print("\nPrevisões:")
print(media_prev_SVR.rename("previsto"))

# ---------------------------------------------------------------
# Ponto de quebra estrutural a destacar no gráfico
# ---------------------------------------------------------------
# Opção A: você já tem uma série dummy de quebra (1 no período de quebra, 0 no resto)
# Break = df["Break"]              # <-- Opção A: descomente se tiver a dummy
data_quebra_manual = data_quebra          # <-- Opção B: ex. pd.Timestamp("2015-01-01")



# Plot: observado x ajustado x previsão, com destaque da quebra
# ---------------------------------------------------------------
plt.figure(figsize=(12, 6))

# 1. Série original (observada)
plt.plot(y.index, y, label="Total (mil)", color="steelblue", alpha=0.6, linewidth=2)

# 2. Valores ajustados pelo modelo no período de treino
plt.plot(
    valores_ajustados_SVR.index,
    valores_ajustados_SVR,
    label="Fitted Values",
    color="orange",
    linestyle="--",
)

# 3. Previsão futura
plt.plot(media_prev_SVR.index, media_prev_SVR, label="Forecasting", color="#d62728", linewidth=2)

# Conexão visual entre o último dado real e o início da previsão (evita "buraco" no gráfico)
ultima_data = y.index[-1]
ultimo_valor = y.iloc[-1]
plt.plot(
    [ultima_data, media_prev_SVR.index[0]],
    [ultimo_valor, media_prev_SVR.iloc[0]],
    color="#d62728",
    linewidth=2,
)

# 4. Marcação visual da quebra estrutural
try:
    if data_quebra_manual is not None:
        data_quebra = data_quebra_manual
    else:
        data_quebra = Break[Break == 1].index[0]
    plt.axvline(x=data_quebra, color="gray", linestyle="-.", linewidth=1.0, label="Break")
except Exception as e:
    print("Aviso: Não foi possível plotar a linha de quebra.", e)

plt.title("SVR: Training and Forecasting")
plt.xlabel("Year", fontsize=12)
plt.ylabel("CO2 Emissions", fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True, linestyle=":", alpha=0.7)
plt.tight_layout()
plt.show()

#%%
# =========================================================================
# 15. CONSOLIDAÇÃO DAS PREVISÕES E COMPARAÇÃO DE ACURÁCIA (TREINO)
# =========================================================================
print("\nConsolidando resultados e calculando métricas...")

# --- 1) TABELA COM TODAS AS PREVISÕES (6 PASSOS À FRENTE) ---
df_previsoes = pd.DataFrame({
    'SARIMAX': previsao_media_Sarima,
    'Random Walk': previsao_rw,
    'Holt-Winters': previsao_hw,
    'LSTM': previsao_lstm,
    'BSM (Kalman)': media_prev_MEB,
    'SVR': media_prev_SVR
})

print("\n" + "="*70)
print("PREVISÕES FUTURAS (6 Passos à Frente)")
print("="*70)
print(df_previsoes.round(2))

# --- 2) CÁLCULO DAS MÉTRICAS DE ACURÁCIA (TREINO) ---
def calcular_metricas(y_true, y_pred):
    # Concatena alinhando pelo índice e remove os NAs (do shift, lags, etc)
    df_metrics = pd.concat([y_true, y_pred], axis=1).dropna()
    df_metrics.columns = ['y', 'y_hat']
    
    y = df_metrics['y'].values
    y_hat = df_metrics['y_hat'].values
    n = len(y)
    
    # RMSE
    rmse = np.sqrt(np.mean((y - y_hat)**2))
    
    # MAPE 
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y - y_hat) / y)) * 100
        
    # TIC (Theil's Inequality Coefficient)
    numerador_tic = np.sqrt(np.mean((y - y_hat)**2))
    denominador_tic = np.sqrt(np.mean(y**2)) + np.sqrt(np.mean(y_hat**2))
    tic = numerador_tic / denominador_tic if denominador_tic != 0 else np.nan
    
    return {'MAPE (%)': mape, 'RMSE': rmse, 'TIC': tic}

# Dicionário organizando todos os ajustes (excluindo os primeiros dados nulos de treino)
dicionario_ajustes = {
    'SARIMAX': valores_ajustados_sarimax[1:], 
    'Random Walk': ajuste_rw,
    'Holt-Winters': valores_ajustados_hw,
    'LSTM': valores_ajustados_LSTM,
    'BSM (Kalman)': valores_ajustados_meb[1:],
    'SVR': valores_ajustados_SVR
}

# Gerando a tabela de acurácia
resultados_metricas = []
for nome_modelo, fitted_vals in dicionario_ajustes.items():
    metricas = calcular_metricas(serie_alvo, fitted_vals)
    metricas['Modelo'] = nome_modelo
    resultados_metricas.append(metricas)

df_acuracia = pd.DataFrame(resultados_metricas)
df_acuracia.set_index('Modelo', inplace=True)
df_acuracia = df_acuracia.sort_values(by='MAPE (%)')

print("\n" + "="*70)
print("MÉTRICAS DE ACURÁCIA (DADOS DE TREINAMENTO)")
print("="*70)
print(df_acuracia.round(4))

#%%
# =========================================================================
# 16. DIAGNÓSTICO DOS RESÍDUOS (GRÁFICOS E TESTES DE NORMALIDADE)
# =========================================================================
print("\n" + "="*70)
print("INICIANDO DIAGNÓSTICO DOS RESÍDUOS...")
print("="*70)

# Função para replicar o plot_diagnostics() em qualquer modelo
def plot_diagnostics_custom(resid, title):
    # Padronizando os resíduos (média 0, desvio padrão 1) para o gráfico
    resid_std = (resid - resid.mean()) / resid.std()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Diagnóstico de Resíduos: {title}', fontsize=16, fontweight='bold')

    # 1. Resíduos Padronizados no Tempo
    axes[0, 0].plot(resid_std, color='steelblue')
    axes[0, 0].axhline(0, color='black', linestyle='--', alpha=0.7)
    axes[0, 0].set_title('Resíduos Padronizados')
    
    # 2. Histograma e Densidade
    sns.histplot(resid_std, kde=True, stat='density', ax=axes[0, 1], color='steelblue')
    # Curva Normal N(0,1) teórica para comparar
    x = np.linspace(resid_std.min()-1, resid_std.max()+1, 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, 0, 1), 'k', linewidth=2, label='Normal Dist (N(0,1))')
    axes[0, 1].set_title('Histograma e Densidade KDE')
    axes[0, 1].legend()

    # 3. Normal Q-Q Plot
    # line='45' compara com a normal padrão, ideal já que padronizamos os resíduos
    qqplot(resid_std, line='45', ax=axes[1, 0], color='steelblue', alpha=0.7)
    axes[1, 0].set_title('Normal Q-Q')

    # 4. Correlograma (ACF)
    plot_acf(resid, ax=axes[1, 1], lags=20, color='steelblue')
    axes[1, 1].set_title('Correlograma (ACF)')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90) # Espaço para o título geral
    plt.show()

# Função para executar os testes estatísticos
def testes_normalidade(resid, nome_modelo):
    print(f"\n--- TESTES DE NORMALIDADE: {nome_modelo.upper()} ---")
    alpha = 0.05
    
    # 1. Teste de Shapiro-Wilk
    sw_stat, sw_p_value = stats.shapiro(resid)
    print(f"[Shapiro-Wilk] Estatística W: {sw_stat:.4f} | P-valor: {sw_p_value:.4f}")
    if sw_p_value > alpha:
        print(" -> Conclusão (SW): Não rejeitamos H0 (Resíduos parecem normais).")
    else:
        print(" -> Conclusão (SW): Rejeitamos H0 (Resíduos NÃO são normais).")

    # 2. Teste de Anderson-Darling
    ad_resultado = stats.anderson(resid, dist='norm')
    print(f"[Anderson-Darling] Estatística: {ad_resultado.statistic:.4f}")
    
    # Pegando o valor crítico para 5% (índice 2 no array do scipy)
    indice_5pct = np.where(ad_resultado.significance_level == 5.0)[0][0]
    valor_critico_5pct = ad_resultado.critical_values[indice_5pct]
    print(f" -> Valor crítico (5%): {valor_critico_5pct:.4f}")
    
    if ad_resultado.statistic < valor_critico_5pct:
        print(" -> Conclusão (AD): Não rejeitamos H0 (Resíduos parecem normais).")
    else:
        print(" -> Conclusão (AD): Rejeitamos H0 (Resíduos NÃO são normais).")
    print("-" * 55)


# Loop iterando por todos os modelos que você consolidou na seção 15
# (Utilizando o dicionario_ajustes já criado no final do script anterior)
for nome_modelo, fitted_vals in dicionario_ajustes.items():
    # 1. Calcula o resíduo do modelo alinhando a série real com a predita (removendo NaNs)
    df_resid = pd.concat([serie_alvo, fitted_vals], axis=1).dropna()
    df_resid.columns = ['Real', 'Ajustado']
    residuos_modelo = df_resid['Real'] - df_resid['Ajustado']
    
    # 2. Exibe os Gráficos de Diagnóstico
    plot_diagnostics_custom(residuos_modelo, nome_modelo)
    
    # 3. Roda os testes de Normalidade
    testes_normalidade(residuos_modelo, nome_modelo)

print("\nAnálise de resíduos concluída com sucesso!")



