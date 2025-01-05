# PROYECTO FINAL
# Realizado por: Maria Elena Guevara
import streamlit as st
import pandas as pd
import plotly.express as px
import statsmodels.api as sm

# Título de la aplicación
st.title("Generador de Gráficos, Modelos Descriptivos y Análisis de Regresión")

# Subir archivo (acepta CSV, XLS y XLSX)
uploaded_file = st.file_uploader("Sube un archivo (CSV, XLS o XLSX)", type=["csv", "xls", "xlsx"])

# Si el archivo es cargado
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    try:
        # Leer el archivo según su tipo
        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xls', 'xlsx']:
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Tipo de archivo no soportado.")
            df = None

        # Describir la estructura del archivo t
        if st.checkbox("Mostrar descripción del contenido de cada columna"):
            # Generar una descripción del contenido de cada columna
            column_descriptions = []
            for col in df.columns:
                column_info = {
                    "Columna": col,
                    "Tipo de Datos": df[col].dtype,
                    "Longitud Máxima (caracteres)": df[col].astype(str).str.len().max()
                }
                column_descriptions.append(column_info)  
    
            # Crear un DataFrame para presentar los datos
            column_details_df = pd.DataFrame(column_descriptions)
    
            st.write("Descripción del contenido de cada columna:")
            st.write(column_details_df)
        
        # Verificar si el DataFrame se cargó correctamente
        if df is not None and not df.empty:
            st.subheader("Vista previa del archivo cargado")
            st.dataframe(df.head())

            # Análisis exploratorio
            st.subheader("Análisis Exploratorio")
            if st.checkbox("Mostrar estadísticas descriptivas"):
                st.write(df.describe(include=['number']))
               

            # Filtrar columnas por tipo de analisis  
            text_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            datetime_columns = df.select_dtypes(include=['datetime64']).columns.tolist()

            # Gráficos de dispersión
            st.subheader("Gráficos de Dispersión")
            x_axis_columns = st.multiselect("Selecciona las columnas para el eje X (Texto)", text_columns)
            y_axis_columns = st.multiselect("Selecciona las columnas para el eje Y (Numérico)", numeric_columns)

            if x_axis_columns and y_axis_columns:
                for x_col in x_axis_columns:
                    for y_col in y_axis_columns:
                        fig = px.scatter(df, x=x_col, y=y_col, title=f"Gráfico de {y_col} vs {x_col}")
                        st.plotly_chart(fig)

            # Estadísticas descriptivas por categorías
            st.subheader("Modelos Descriptivos")
            selected_models = st.multiselect(
                "Selecciona los modelos descriptivos para el análisis",
                ["Media", "Mediana", "Desviación estándar", "Rango"]
            )

            if selected_models:
                group_cols = x_axis_columns if x_axis_columns else numeric_columns
                for model in selected_models:
                    if model == "Media":
                        result = df.groupby(group_cols)[y_axis_columns].mean()
                        st.write("Media:")
                        st.write(result)
                        st.plotly_chart(px.bar(result, title="Media por categoría"))
                    elif model == "Mediana":
                        result = df.groupby(group_cols)[y_axis_columns].median()
                        st.write("Mediana:")
                        st.write(result)
                        st.plotly_chart(px.bar(result, title="Mediana por categoría"))
                    elif model == "Desviación estándar":
                        result = df.groupby(group_cols)[y_axis_columns].std()
                        st.write("Desviación estándar:")
                        st.write(result)
                        st.plotly_chart(px.bar(result, title="Desviación estándar por categoría"))
                    elif model == "Rango":
                        result = df.groupby(group_cols)[y_axis_columns].apply(lambda x: x.max() - x.min())
                        st.write("Rango:")
                        st.write(result)
                        st.plotly_chart(px.bar(result, title="Rango por categoría"))

            # Análisis de Regresión
            st.subheader("Análisis de Regresión")
            regression_type = st.selectbox(
                "Selecciona el tipo de regresión",
                ["Regresión Lineal Simple", "Análisis de Series Temporales", "Análisis Multivariables"]
            )

            if regression_type == "Regresión Lineal Simple":
                regression_x_column = st.selectbox("Selecciona la columna para el eje X (numérica)", numeric_columns)
                regression_y_column = st.selectbox("Selecciona la columna para el eje Y (numérica)", numeric_columns)

                if regression_x_column and regression_y_column and regression_x_column != regression_y_column:
                    st.write(f"**Regresión Lineal Simple entre {regression_x_column} y {regression_y_column}**")
                    X = df[regression_x_column]
                    Y = df[regression_y_column]
                    X = sm.add_constant(X)
                    model = sm.OLS(Y, X).fit()
                    st.write(model.summary())
                    
                    # Interpretación de los resultados a nivel de regresion simple 
                    st.subheader("Interpretación de los resultados:")
                    st.write(f"- **Coeficiente de {regression_x_column}**: {model.params[regression_x_column]:.4f}, lo que indica que un aumento de una unidad en `{regression_x_column}` se asocia con un cambio promedio de {model.params[regression_x_column]:.4f} unidades en `{regression_y_column}`.")
                    st.write(f"- **Intercepto**: {model.params['const']:.4f}, que representa el valor promedio de `{regression_y_column}` cuando `{regression_x_column}` es cero.")
                    st.write(f"- **R-cuadrado**: {model.rsquared:.4f}, lo que sugiere que el modelo explica el {model.rsquared * 100:.2f}% de la variabilidad observada en `{regression_y_column}`.")
                    st.write(f"- **Valor p**: {model.pvalues[regression_x_column]:.4e}, {'significativo' if model.pvalues[regression_x_column] < 0.05 else 'no significativo'}, lo que indica si `{regression_x_column}` tiene una relación estadísticamente significativa con `{regression_y_column}`.")

                    fig = px.scatter(df, x=regression_x_column, y=regression_y_column, title="Regresión Lineal")
                    fig.add_scatter(x=df[regression_x_column], y=model.predict(X), mode='lines', name='Línea de Regresión')
                    st.plotly_chart(fig)

            elif regression_type == "Análisis de Series Temporales":
                if datetime_columns:
                    time_column = st.selectbox("Selecciona la columna de fecha/tiempo", datetime_columns)
                    value_column = st.selectbox("Selecciona la columna de valores (numérica)", numeric_columns)

                    if time_column and value_column:
                        st.write(f"**Análisis de Series Temporales para {value_column} respecto a {time_column}**")
                        df_sorted = df.sort_values(by=time_column)
                        fig = px.line(df_sorted, x=time_column, y=value_column, title="Serie Temporal")
                        st.plotly_chart(fig)

                        # Interpretación de los resultados
                        st.subheader("Interpretación de los resultados:")
                        st.write(f"- La serie temporal muestra cómo `{value_column}` cambia a lo largo del tiempo.")
                        st.write("- Observa si existen tendencias (aumento/disminución), estacionalidad (patrones recurrentes), o irregularidades.")
                        st.write("- Para un análisis más detallado, se pueden aplicar descomposición estacional o modelos ARIMA para pronósticos.")

                else:
                    st.warning("No se encontraron columnas de tipo fecha/tiempo en el archivo.")

            elif regression_type == "Análisis Multivariables":
                independent_vars = st.multiselect("Selecciona las variables independientes (X)", numeric_columns)
                dependent_var = st.selectbox("Selecciona la variable dependiente (Y)", numeric_columns)

                if independent_vars and dependent_var and dependent_var not in independent_vars:
                    st.write(f"**Regresión Multivariables para predecir {dependent_var} usando {', '.join(independent_vars)}**")
                    X = df[independent_vars]
                    Y = df[dependent_var]
                    X = sm.add_constant(X)
                    model = sm.OLS(Y, X).fit()
                    st.write(model.summary()) 

                    # Interpretación de los resultados a nivel de Multivariables
                    st.subheader("Interpretación de los resultados:")
                    st.write(f"- **R-cuadrado**: {model.rsquared:.4f}, lo que indica que el modelo explica el {model.rsquared * 100:.2f}% de la variabilidad en `{dependent_var}`.")
                    st.write("- **Coeficientes de las variables independientes**:")
                    for var in independent_vars:
                        coef = model.params[var]
                        pval = model.pvalues[var]
                        st.write(f"  - `{var}`: Coeficiente = {coef:.4f}, Valor p = {pval:.4e} {'(Significativo)' if pval < 0.05 else '(No significativo)'}")
                    st.write(f"- **Intercepto**: {model.params['const']:.4f}, representando el valor esperado de `{dependent_var}` cuando todas las variables independientes son cero.")

        else:
            st.error("El archivo está vacío o no se cargó correctamente.")
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")