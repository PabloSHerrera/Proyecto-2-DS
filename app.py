import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib  # Para cargar los modelos entrenados

# Título de la aplicación
st.title('Predicción de Precios de Commodities')

# Cargar los modelos entrenados
rf_model = joblib.load('random_forest_model.pkl')  # Asegúrate de tener el modelo guardado como random_forest_model.pkl
xgb_model = joblib.load('xgboost_model.pkl')  # Carga el modelo XGBoost
lstm_model = joblib.load('lstm_model.pkl')  # Carga el modelo LSTM

# Función para predecir con el modelo seleccionado
def predict_with_model(model_name, data):
    if model_name == "Random Forest":
        model = rf_model
    elif model_name == "XGBoost":
        model = xgb_model
    elif model_name == "LSTM":
        model = lstm_model
    else:
        return "Modelo no reconocido"

    predictions = model.predict(data)
    return predictions

# Cargar los datos
st.sidebar.header('Cargar Datos')
uploaded_file = st.sidebar.file_uploader("Sube el archivo CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Datos cargados exitosamente.")
    st.dataframe(data.head())  # Muestra las primeras filas de los datos

# Análisis exploratorio de los datos (EDA)
st.sidebar.header('Análisis Exploratorio de Datos')

# Mostrar estadísticas descriptivas
if st.sidebar.checkbox('Mostrar estadísticas descriptivas'):
    st.write(data.describe())

# Correlación entre las variables
if st.sidebar.checkbox('Mostrar matriz de correlación'):
    corr = data.corr()
    st.write(corr)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Gráficos interactivos
if st.sidebar.checkbox('Mostrar gráfico interactivo de dispersión'):
    if 'columna_x' in data.columns and 'columna_y' in data.columns:  # Asegúrate de que estas columnas existan
        fig = px.scatter(data, x='columna_x', y='columna_y', color='columna_clasificación')
        st.plotly_chart(fig)
    else:
        st.write("Las columnas 'columna_x' o 'columna_y' no existen en el dataset.")

# Barra lateral para seleccionar el modelo
model_option = st.sidebar.selectbox("Selecciona el modelo", ['Random Forest', 'XGBoost', 'LSTM'])

# Al realizar la predicción
if st.sidebar.button('Realizar Predicción'):
    if uploaded_file is not None:
        # Preprocesamiento (si lo necesitas, aquí aplicas la normalización, manejo de valores nulos, etc.)
        # Por ejemplo:
        # data_scaled = scaler.transform(data)  # Si usas un scaler, por ejemplo, StandardScaler

        # Realizar la predicción
        predictions = predict_with_model(model_option, data)
        st.write(f"Predicciones con el modelo {model_option}:")
        st.write(predictions)
        
        # Mostrar gráfico de las predicciones
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(data['Fecha'], predictions)  # Asegúrate de tener una columna 'Fecha'
        ax.set_title(f'Predicciones de Precios de Commodities con {model_option}')
        st.pyplot(fig)

# Mostrar el rendimiento de los modelos
if st.sidebar.checkbox('Ver rendimiento de los modelos'):
    st.write("Rendimiento de los modelos:")
    # Aquí puedes mostrar el rendimiento de los diferentes modelos usando RMSE, precisión, etc.
    # Por ejemplo, podrías mostrar los valores de RMSE para cada modelo:
    st.write("Random Forest RMSE: 0.026")
    st.write("XGBoost RMSE: 0.022")
    st.write("LSTM RMSE: 0.030")

# **Opción para cambiar parámetros del modelo (ajuste de hiperparámetros)**
st.sidebar.header('Ajuste de Parámetros')
n_estimators = st.sidebar.slider('Número de estimadores (Random Forest)', min_value=50, max_value=300, value=100, step=50)
max_depth = st.sidebar.slider('Profundidad máxima (Random Forest)', min_value=5, max_value=30, value=10, step=5)

st.sidebar.write(f'Parámetros seleccionados: {n_estimators} estimadores, {max_depth} profundidad máxima')

# Función para ajustar el modelo (puedes agregar ajustes según lo necesario)
def adjust_model(data):
    rf_model.set_params(n_estimators=n_estimators, max_depth=max_depth)
    rf_model.fit(data.drop('target', axis=1), data['target'])
    return rf_model

if st.sidebar.button('Ajustar Modelo'):
    if uploaded_file is not None:
        # Realizar el ajuste del modelo con los nuevos parámetros
        adjusted_model = adjust_model(data)
        st.write("Modelo ajustado exitosamente con los parámetros seleccionados.")
        st.write(f"Modelo con {n_estimators} estimadores y profundidad máxima de {max_depth}.")

