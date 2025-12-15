import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
import geopandas as gpd
import pandas as pd

class GeospatialDatabase:
    def __init__(self):
        self.db_user = os.getenv('DB_USER', 'postgres')
        self.db_password = os.getenv('DB_PASSWORD', 'postgres')
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = os.getenv('DB_PORT', '5432')
        self.db_name = os.getenv('DB_NAME', 'geospatial_db')
        
        self.connection_url = f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
        self.engine = create_engine(self.connection_url)
    
    def execute_query(self, query, params=None):
        """Executa uma query SQL"""
        with self.engine.connect() as conn:
            if params:
                result = conn.execute(text(query), params)
            else:
                result = conn.execute(text(query))
            return result.fetchall()
    
    def import_geodata(self, gdf, table_name, if_exists='replace'):
        """Importa GeoDataFrame para o PostGIS"""
        gdf.to_postgis(
            table_name,
            self.engine,
            if_exists=if_exists,
            index=False
        )
        print(f"Dados importados para a tabela {table_name}")
    
    def get_geodata(self, table_name, columns='*', where=None):
        """Recupera dados geográficos do PostGIS"""
        query = f"SELECT {columns} FROM {table_name}"
        if where:
            query += f" WHERE {where}"
        
        return gpd.read_postgis(query, self.engine, geom_col='geometry')
    
    def create_spatial_index(self, table_name, geometry_column='geometry'):
        """Cria índice espacial"""
        query = f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_geometry 
        ON {table_name} USING GIST ({geometry_column});
        """
        self.execute_query(query)
    
    def spatial_query(self, table_name, geometry_wkt, operation='ST_Intersects'):
        """Realiza consulta espacial"""
        query = f"""
        SELECT * FROM {table_name}
        WHERE {operation}(geometry, ST_GeomFromText(:geom, 4326))
        """
        return self.get_geodata(query, {'geom': geometry_wkt})