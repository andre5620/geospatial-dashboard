import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium, folium_static
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importar módulos do projeto
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from api_client import OpenDataAPIClient
from database import GeospatialDatabase
from esda_analysis import ExploratorySpatialDataAnalysis
from variogram_model import VariogramModeler

# Configuração da página
st.set_page_config(
    page_title="Dashboard Geoespacial",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🌍 Dashboard de Análise Geoespacial")
st.markdown("""
Este dashboard realiza análise exploratória de dados espaciais (ESDA) e modelagem de semivariogramas.
Os dados são armazenados no PostGIS e processados em tempo real.
""")

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Seção de fonte de dados
    st.subheader("Fonte de Dados")
    data_source = st.selectbox(
        "Selecione a fonte de dados:",
        ["USGS Earthquakes", "OpenAQ Air Quality", "Upload de Arquivo", "Banco de Dados"]
    )
    
    if data_source == "USGS Earthquakes":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Data inicial", 
                                      datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("Data final", datetime.now())
        
        min_magnitude = st.slider("Magnitude mínima", 0.0, 10.0, 2.5, 0.1)
    
    elif data_source == "OpenAQ Air Quality":
        city = st.text_input("Cidade (opcional)", "")
    
    elif data_source == "Upload de Arquivo":
        uploaded_file = st.file_uploader(
            "Escolha um arquivo",
            type=['geojson', 'shp', 'csv', 'gpkg']
        )
    
    # Seção de análise
    st.subheader("Configurações de Análise")
    analysis_type = st.multiselect(
        "Selecione as análises:",
        ["ESDA Básica", "Autocorrelação Espacial", "Hot/Cold Spots", 
         "Variograma", "Krigagem"],
        default=["ESDA Básica", "Autocorrelação Espacial"]
    )
    
    if "Variograma" in analysis_type:
        variogram_model = st.selectbox(
            "Modelo de Variograma:",
            ["spherical", "exponential", "gaussian"]
        )
    
    # Botão para executar análise
    run_analysis = st.button("🚀 Executar Análise", type="primary")

# Inicializar sessão
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'db' not in st.session_state:
    st.session_state.db = GeospatialDatabase()

# Função principal para carregar dados
def load_data():
    with st.spinner("Carregando dados..."):
        try:
            api_client = OpenDataAPIClient()
            
            if data_source == "USGS Earthquakes":
                gdf = api_client.fetch_earthquakes(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d'),
                    min_magnitude
                )
                if 'mag' in gdf.columns:
                    value_column = 'mag'
                else:
                    value_column = gdf.select_dtypes(include=[np.number]).columns[0]
            
            elif data_source == "OpenAQ Air Quality":
                gdf = api_client.fetch_air_quality(city if city else None)
                value_column = 'value'
            
            elif data_source == "Upload de Arquivo" and uploaded_file:
                if uploaded_file.name.endswith('.geojson'):
                    gdf = gpd.read_file(uploaded_file)
                elif uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                    # Supor que tenha colunas lat/lon
                    gdf = gpd.GeoDataFrame(
                        df,
                        geometry=gpd.points_from_xy(df.longitude, df.latitude),
                        crs='EPSG:4326'
                    )
                value_column = st.selectbox(
                    "Selecione a coluna numérica para análise:",
                    gdf.select_dtypes(include=[np.number]).columns.tolist()
                )
            
            else:
                st.warning("Por favor, selecione uma fonte de dados válida.")
                return None, None
            
            # Salvar no banco de dados
            if len(gdf) > 0:
                st.session_state.db.import_geodata(gdf, 'analysis_data')
                st.session_state.db.create_spatial_index('analysis_data')
                st.success(f"Dados carregados: {len(gdf)} registros")
            
            return gdf, value_column
        
        except Exception as e:
            st.error(f"Erro ao carregar dados: {str(e)}")
            return None, None

# Executar análise quando o botão for pressionado
if run_analysis:
    gdf, value_column = load_data()
    if gdf is not None:
        st.session_state.gdf = gdf
        st.session_state.value_column = value_column

# Mostrar dados se carregados
if st.session_state.gdf is not None:
    gdf = st.session_state.gdf
    value_column = st.session_state.value_column
    
    # Layout principal
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visualização", 
        "🔍 Análise ESDA", 
        "📈 Variograma",
        "💾 Banco de Dados"
    ])
    
    with tab1:
        # Mapa interativo
        st.subheader("Mapa Interativo")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Criar mapa folium
            m = folium.Map(
                location=[gdf.geometry.centroid.y.mean(), 
                         gdf.geometry.centroid.x.mean()],
                zoom_start=10,
                control_scale=True
            )
            
            # Adicionar pontos ao mapa
            for idx, row in gdf.iterrows():
                popup_text = f"""
                <b>Valor:</b> {row.get(value_column, 'N/A')}<br>
                <b>Coordenadas:</b> {row.geometry.y:.4f}, {row.geometry.x:.4f}
                """
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=8,
                    popup=folium.Popup(popup_text, max_width=300),
                    color='blue' if row.get(value_column, 0) >= gdf[value_column].median() else 'red',
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)
            
            # Exibir mapa
            st_folium(m, width=800, height=500)
        
        with col2:
            # Estatísticas básicas
            st.metric("Total de Registros", len(gdf))
            st.metric("Média", f"{gdf[value_column].mean():.2f}")
            st.metric("Mediana", f"{gdf[value_column].median():.2f}")
            st.metric("Desvio Padrão", f"{gdf[value_column].std():.2f}")
            
            # Histograma
            fig = px.histogram(gdf, x=value_column, nbins=30,
                              title=f"Distribuição de {value_column}")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if "ESDA Básica" in analysis_type or "Autocorrelação Espacial" in analysis_type:
            st.subheader("Análise Exploratória de Dados Espaciais (ESDA)")
            
            # Inicializar análise ESDA
            esda = ExploratorySpatialDataAnalysis(gdf, value_column)
            
            # Criar pesos espaciais
            with st.spinner("Calculando pesos espaciais..."):
                weights = esda.create_spatial_weights()
            
            # Métricas globais
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                moran = esda.global_moran_i()
                st.metric("I de Moran", f"{moran['I']:.3f}")
                st.caption(f"p-value: {moran['p_value']:.4f}")
            
            with col2:
                geary = esda.geary_c()
                st.metric("C de Geary", f"{geary['C']:.3f}")
                st.caption(f"p-value: {geary['p_value']:.4f}")
            
            with col3:
                getis_ord = esda.getis_ord_g()
                st.metric("G de Getis-Ord", f"{getis_ord['G']:.3f}")
            
            with col4:
                st.metric("Autocorrelação", 
                         "Significativa" if moran['p_value'] < 0.05 else "Não Significativa")
            
            # Gráficos ESDA
            if "Autocorrelação Espacial" in analysis_type:
                st.subheader("Análise de Autocorrelação Espacial")
                
                fig = esda.plot_spatial_autocorrelation()
                st.pyplot(fig)
            
            if "Hot/Cold Spots" in analysis_type:
                st.subheader("Hot Spots e Cold Spots")
                
                hot_cold_gdf = esda.hot_cold_spots()
                
                # Mapa de hot/cold spots
                fig, ax = plt.subplots(figsize=(10, 8))
                hot_cold_gdf.plot(
                    column='hot_cold_spot',
                    categorical=True,
                    legend=True,
                    cmap='RdYlBu_r',
                    ax=ax,
                    edgecolor='black'
                )
                ax.set_title("Hot Spots e Cold Spots (Getis-Ord Gi*)")
                ax.set_axis_off()
                st.pyplot(fig)
                
                # Tabela de estatísticas
                spot_counts = hot_cold_gdf['hot_cold_spot'].value_counts()
                st.dataframe(spot_counts)
    
    with tab3:
        if "Variograma" in analysis_type:
            st.subheader("Modelagem de Semivariograma")
            
            # Inicializar modelador de variograma
            variogram_modeler = VariogramModeler(gdf, value_column)
            
            # Calcular variograma empírico
            with st.spinner("Calculando variograma empírico..."):
                distances, semivariances, pair_counts = variogram_modeler.calculate_empirical_variogram(
                    max_distance=st.slider("Distância máxima", 0.1, 100.0, 50.0),
                    n_bins=st.slider("Número de bins", 5, 30, 15)
                )
            
            # Ajustar modelo
            with st.spinner(f"Ajustando modelo {variogram_model}..."):
                model_result = variogram_modeler.fit_variogram_model(
                    model_type=variogram_model
                )
            
            if model_result:
                # Mostrar parâmetros do modelo
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Nugget", f"{model_result['parameters']['nugget']:.3f}")
                with col2:
                    st.metric("Sill", f"{model_result['parameters']['sill']:.3f}")
                with col3:
                    st.metric("Range", f"{model_result['parameters']['range']:.3f}")
                with col4:
                    st.metric("R²", f"{model_result['r_squared']:.3f}")
                
                # Plotar variograma
                fig = variogram_modeler.plot_variogram(model_result)
                st.pyplot(fig)
                
                # Validação cruzada
                if st.checkbox("Realizar validação cruzada"):
                    with st.spinner("Realizando validação cruzada..."):
                        cv_results = variogram_modeler.cross_validation(model_result)
                    
                    st.metric("MSE Médio (Validação Cruzada)", 
                             f"{cv_results['mean_mse']:.3f}")
                    st.metric("Desvio Padrão do MSE", 
                             f"{cv_results['std_mse']:.3f}")
            
            # Comparação de modelos
            if st.checkbox("Comparar diferentes modelos"):
                models = ['spherical', 'exponential', 'gaussian']
                results = []
                
                for model_type in models:
                    result = variogram_modeler.fit_variogram_model(model_type)
                    if result:
                        results.append({
                            'Modelo': model_type,
                            'Nugget': result['parameters']['nugget'],
                            'Sill': result['parameters']['sill'],
                            'Range': result['parameters']['range'],
                            'R²': result['r_squared']
                        })
                
                if results:
                    comparison_df = pd.DataFrame(results)
                    st.dataframe(comparison_df.style.highlight_max(subset=['R²']))
    
    with tab4:
        st.subheader("Gerenciamento do Banco de Dados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Consultar tabelas
            st.subheader("Consultar Dados")
            query = st.text_area("SQL Query", 
                                "SELECT * FROM analysis_data LIMIT 10")
            
            if st.button("Executar Query"):
                try:
                    results = st.session_state.db.execute_query(query)
                    if results:
                        df = pd.DataFrame(results)
                        st.dataframe(df)
                except Exception as e:
                    st.error(f"Erro na query: {str(e)}")
        
        with col2:
            # Exportar dados
            st.subheader("Exportar Dados")
            
            export_format = st.selectbox(
                "Formato de exportação",
                ["GeoJSON", "CSV", "Shapefile", "Parquet"]
            )
            
            if st.button("Exportar Dados"):
                try:
                    if export_format == "GeoJSON":
                        gdf.to_file("exported_data.geojson", driver='GeoJSON')
                    elif export_format == "CSV":
                        # Exportar sem geometria
                        df = pd.DataFrame(gdf.drop(columns='geometry'))
                        df.to_csv("exported_data.csv", index=False)
                    
                    st.success(f"Dados exportados como {export_format}")
                    
                    # Botão de download
                    with open(f"exported_data.{export_format.lower()}", "rb") as f:
                        st.download_button(
                            "📥 Download",
                            f,
                            file_name=f"exported_data.{export_format.lower()}"
                        )
                
                except Exception as e:
                    st.error(f"Erro ao exportar: {str(e)}")
        
        # Estatísticas do banco
        st.subheader("Estatísticas do Banco de Dados")
        
        try:
            # Consultar estatísticas
            stats_query = """
            SELECT 
                COUNT(*) as total_records,
                MIN(ST_X(geometry)) as min_lon,
                MAX(ST_X(geometry)) as max_lon,
                MIN(ST_Y(geometry)) as min_lat,
                MAX(ST_Y(geometry)) as max_lat
            FROM analysis_data
            """
            
            stats = st.session_state.db.execute_query(stats_query)
            if stats:
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Registros", stats[0][0])
                with col2:
                    st.metric("Min Lon", f"{stats[0][1]:.4f}")
                with col3:
                    st.metric("Max Lon", f"{stats[0][2]:.4f}")
                with col4:
                    st.metric("Min Lat", f"{stats[0][3]:.4f}")
                with col5:
                    st.metric("Max Lat", f"{stats[0][4]:.4f}")
        
        except Exception as e:
            st.warning(f"Não foi possível obter estatísticas: {str(e)}")

else:
    # Página inicial quando não há dados
    st.info("👈 Configure os parâmetros na sidebar e clique em 'Executar Análise' para começar.")
    
    # Exemplos de APIs públicas
    st.subheader("📡 APIs Públicas Disponíveis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **USGS Earthquakes**
        - Dados de terremotos em tempo real
        - Frequência: atualização contínua
        - Formato: GeoJSON
        """)
    
    with col2:
        st.markdown("""
        **OpenAQ Air Quality**
        - Dados de qualidade do ar
        - 100+ países
        - Parâmetros: PM2.5, O₃, NO₂, etc.
        """)
    
    with col3:
        st.markdown("""
        **Outras Fontes**
        - Dados climáticos
        - Informações demográficas
        - Dados de tráfego
        - Sensoriamento remoto
        """)

# Rodapé
st.markdown("---")
st.markdown("""
**📚 Tecnologias utilizadas:** Streamlit, PostGIS, Docker, Python, GeoPandas, PySAL

**🔧 Desenvolvido por:** André Luiz | [GitHub](https://github.com/andre5620)
""")