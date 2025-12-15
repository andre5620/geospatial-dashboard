import geopandas as gpd
import pandas as pd
from sqlalchemy import create_engine
import os

def import_sample_data():
    """Importa dados de exemplo para o banco"""
    
    # Conectar ao banco
    engine = create_engine('postgresql://postgres:postgres@localhost:5432/geospatial_db')
    
    # Criar dados de exemplo (pontos aleatórios com valores)
    import numpy as np
    np.random.seed(42)
    
    n_points = 100
    lat = np.random.uniform(-23.8, -23.5, n_points)
    lon = np.random.uniform(-46.8, -46.5, n_points)
    
    # Criar valores correlacionados espacialmente
    values = []
    for i in range(n_points):
        # Valor base mais ruído
        base_value = 50 + 20 * np.sin(lat[i] * 10) + 30 * np.cos(lon[i] * 10)
        noise = np.random.normal(0, 5)
        values.append(max(0, base_value + noise))
    
    # Criar GeoDataFrame
    data = {
        'id': range(n_points),
        'value': values,
        'category': np.random.choice(['A', 'B', 'C'], n_points),
        'latitude': lat,
        'longitude': lon
    }
    
    df = pd.DataFrame(data)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs='EPSG:4326'
    )
    
    # Importar para o PostGIS
    gdf.to_postgis('sample_data', engine, if_exists='replace', index=False)
    print(f"Dados de exemplo importados: {len(gdf)} registros")
    
    # Criar índice espacial
    with engine.connect() as conn:
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sample_data_geometry 
            ON sample_data USING GIST (geometry);
        """)
    
    print("Índice espacial criado")

if __name__ == "__main__":
    import_sample_data()