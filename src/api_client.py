import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import json

class GeoAPIClient:
    def __init__(self, base_url=None):
        self.base_url = base_url
    
    def fetch_geojson(self, url):
        """Busca dados GeoJSON de uma API"""
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        if data['type'] == 'FeatureCollection':
            gdf = gpd.GeoDataFrame.from_features(data['features'])
            return gdf
        else:
            raise ValueError("Formato GeoJSON não suportado")
    
    def fetch_csv_with_coords(self, url, lat_col='latitude', lon_col='longitude'):
        """Busca CSV com coordenadas e converte para GeoDataFrame"""
        df = pd.read_csv(url)
        
        # Criar geometria a partir de coordenadas
        geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        return gdf
    
    def fetch_openstreetmap(self, bbox, tags=None):
        """Busca dados do OpenStreetMap via Overpass API"""
        overpass_url = "http://overpass-api.de/api/interpreter"
        
        if tags is None:
            tags = {'building': True}
        
        tag_query = ''
        for key, value in tags.items():
            tag_query += f'["{key}"="{value}"]'
        
        query = f"""
        [out:json];
        (
          node{tag_query}({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
          way{tag_query}({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
          relation{tag_query}({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        );
        out body;
        >;
        out skel qt;
        """
        
        response = requests.get(overpass_url, params={'data': query})
        data = response.json()
        
        # Converter para GeoDataFrame (simplificado)
        features = []
        for element in data['elements']:
            if element['type'] == 'node':
                geom = Point(element['lon'], element['lat'])
                features.append({
                    'geometry': geom,
                    'tags': element.get('tags', {}),
                    'id': element['id']
                })
        
        return gpd.GeoDataFrame(features, crs='EPSG:4326')

# Exemplo de uso com API pública
class OpenDataAPIClient(GeoAPIClient):
    def fetch_earthquakes(self, start_date, end_date, min_magnitude=0):
        """Busca dados de terremotos do USGS"""
        url = f"https://earthquake.usgs.gov/fdsnws/event/1/query.geojson"
        params = {
            'starttime': start_date,
            'endtime': end_date,
            'minmagnitude': min_magnitude,
            'orderby': 'time'
        }
        response = requests.get(url, params=params)
        return self.fetch_geojson(response.url)
    
    def fetch_air_quality(self, city=None):
        """Busca dados de qualidade do ar"""
        # Exemplo com API pública
        url = "https://api.openaq.org/v2/latest"
        params = {'limit': 1000}
        if city:
            params['city'] = city
        
        response = requests.get(url, params=params)
        data = response.json()
        
        locations = data['results']
        records = []
        for loc in locations:
            for measurement in loc['measurements']:
                records.append({
                    'city': loc['city'],
                    'location': loc['location'],
                    'parameter': measurement['parameter'],
                    'value': measurement['value'],
                    'unit': measurement['unit'],
                    'latitude': loc['coordinates']['latitude'],
                    'longitude': loc['coordinates']['longitude']
                })
        
        df = pd.DataFrame(records)
        geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
        return gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')