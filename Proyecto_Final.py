import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm

# Título de la aplicación
st.title("Generador de Gráficos, Modelos Descriptivos y Regresión Lineal desde Archivos CSV o Excel")

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

                # Selección de Modelos Descriptivos
                st.subheader("Modelos Descriptivos")
                selected_models = st.multiselect(
                    "Selecciona los modelos descriptivos para el análisis",
                    ["Media", "Mediana", "Desviación estándar", "Rango"]
                )

                # Cálculo de los modelos descriptivos seleccionados
                if "Media" in selected_models:
                    st.write(f"**Media de {', '.join(y_axis_columns)} por categoría de {', '.join(x_axis_columns)}**")
                    media_df = df.groupby(x_axis_columns)[y_axis_columns].mean()
                    st.write(media_df)

                    # Gráfico de barras para la media
                    fig = px.bar(media_df, x=media_df.index, y=media_df.columns, title="Media por categoría")
                    st.plotly_chart(fig)
                
                if "Mediana" in selected_models:
                    st.write(f"**Mediana de {', '.join(y_axis_columns)} por categoría de {', '.join(x_axis_columns)}**")
                    median_df = df.groupby(x_axis_columns)[y_axis_columns].median()
                    st.write(median_df)

                    # Gráfico de barras para la mediana
                    fig = px.bar(median_df, x=median_df.index, y=median_df.columns, title="Mediana por categoría")
                    st.plotly_chart(fig)
                
                if "Desviación estándar" in selected_models:
                    st.write(f"**Desviación estándar de {', '.join(y_axis_columns)} por categoría de {', '.join(x_axis_columns)}**")
                    std_df = df.groupby(x_axis_columns)[y_axis_columns].std()
                    st.write(std_df)

                    # Gráfico de barras para la desviación estándar
                    fig = px.bar(std_df, x=std_df.index, y=std_df.columns, title="Desviación estándar por categoría")
                    st.plotly_chart(fig)
                
                if "Rango" in selected_models:
                    st.write(f"**Rango de {', '.join(y_axis_columns)} por categoría de {', '.join(x_axis_columns)}**")
                    range_df = df.groupby(x_axis_columns)[y_axis_columns].agg(lambda x: x.max() - x.min())
                    st.write(range_df)

                    # Gráfico de barras para el rango
                    fig = px.bar(range_df, x=range_df.index, y=range_df.columns, title="Rango por categoría")
                    st.plotly_chart(fig)

                # Regresión Lineal: Sección independiente
                st.subheader("Análisis de Regresión Lineal")

                # Permitir al usuario seleccionar las columnas nuevas para la regresión
                regression_x_column = st.selectbox("Selecciona la columna para el eje X (numérico) para la regresión", numeric_columns)
                regression_y_column = st.selectbox("Selecciona la columna para el eje Y (numérico) para la regresión", numeric_columns)

                # Verificar que ambas columnas sean numéricas y distintas
                if regression_x_column and regression_y_column and regression_x_column != regression_y_column:
                    # Regresión lineal entre las columnas seleccionadas
                    st.write(f"**Regresión Lineal entre {regression_x_column} y {regression_y_column}**")
                    
                    # Regresión lineal usando statsmodels
                    X = df[regression_x_column]
                    Y = df[regression_y_column]
                    
                    # Añadir una constante a X para el intercepto
                    X = sm.add_constant(X)
                    
                    # Ajustar el modelo de regresresión
                    model = sm.OLS(Y, X).fit()

                    # Mostrar los resultados del modelo
                    st.write(model.summary())

                    # Graficar la línea de regresión sobre el gráfico de dispersión
                    fig = px.scatter(df, x=regression_x_column, y=regression_y_column, title=f"Regresión Lineal: {regression_y_column} vs {regression_x_column}")
                    fig.update_traces(marker=dict(color='blue', size=10))
                    
                    # Añadir la línea de regresión
                    fig.add_scatter(x=df[regression_x_column], y=model.predict(X), mode='lines', name='Línea de Regresión', line=dict(color='red'))
                    
                    st.plotly_chart(fig)

            else:
                st.warning("Por favor selecciona al menos una columna para el eje X y al menos una columna para el eje Y.")
        else:
            st.error("El archivo está vacío o no se ha cargado correctamente.")
    
    except Exception as e:
        # Mostrar un mensaje de error si hay un problema al leer el archivo
        st.error(f"Error al leer el archivo: {str(e)}")
