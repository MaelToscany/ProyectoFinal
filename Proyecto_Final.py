import streamlit as st
import pandas as pd
import plotly.express as px

# Título de la aplicación
st.title("Generador de Gráficos desde Archivos CSV o Excel")

# Subir archivo (acepta CSV, XLS y XLSX)
uploaded_file = st.file_uploader("Sube un archivo (CSV, XLS o XLSX)", type=["csv", "xls", "xlsx"])

# Si el archivo es cargado
if uploaded_file is not None:
    # Obtener la extensión del archivo
    file_extension = uploaded_file.name.split('.')[-1]

    try:
        # Leer el archivo según su tipo (CSV o Excel)
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Tipo de archivo no soportado.")
            df = None

        # Verificar si el dataframe se cargó correctamente
        if df is not None and not df.empty:
            # Mostrar las primeras filas del dataframe para revisar el contenido
            st.subheader("Vista previa del archivo cargado")
            st.dataframe(df.head())  # Muestra las primeras filas del DataFrame

            # Filtrar columnas de tipo texto para el eje X y columnas numéricas para el eje Y
            text_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()  # Eje X (texto)
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()  # Eje Y (numérico)

            # Mostrar las columnas disponibles para el eje X (solo texto) y el eje Y (solo numérico)
            st.subheader("Selecciona las columnas para graficar")

            # Permitir al usuario seleccionar solo las columnas de texto para el eje X
            x_axis_columns = st.multiselect("Selecciona las columnas para el eje X (Texto)", text_columns)

            # Permitir al usuario seleccionar solo las columnas numéricas para el eje Y
            y_axis_columns = st.multiselect("Selecciona las columnas para el eje Y (Numérico)", numeric_columns)

            # Verificar si el usuario ha seleccionado columnas
            if len(x_axis_columns) > 0 and len(y_axis_columns) > 0:
                st.subheader(f"Gráficos de dispersión")

                # Generar gráficos para cada combinación de columnas seleccionadas
                for x_col in x_axis_columns:
                    for y_col in y_axis_columns:
                        # Crear gráfico de dispersión para cada combinación de columnas
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"Gráfico de {y_col} vs {x_col}")
                        st.plotly_chart(fig)
            else:
                st.warning("Por favor selecciona al menos una columna para el eje X (Texto) y al menos una columna para el eje Y (Numérico).")
        else:
            st.error("El archivo está vacío o no se ha cargado correctamente.")
    
    except Exception as e:
        # Mostrar un mensaje de error si hay un problema al leer el archivo
        st.error(f"Error al leer el archivo: {str(e)}")
