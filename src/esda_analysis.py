import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from libpysal.weights import Queen, KNN
from esda.moran import Moran, Moran_Local
from esda.geary import Geary
from esda.getisord import G, G_Local
from splot.esda import plot_moran, lisa_cluster
import contextily as ctx

class ExploratorySpatialDataAnalysis:
    def __init__(self, gdf, value_column):
        self.gdf = gdf
        self.value_column = value_column
        self.weights = None
        
    def create_spatial_weights(self, k=8, queen=True):
        """Cria matriz de pesos espaciais"""
        if queen:
            self.weights = Queen.from_dataframe(self.gdf)
        else:
            self.weights = KNN.from_dataframe(self.gdf, k=k)
        
        # Normalizar pesos
        self.weights.transform = 'r'
        return self.weights
    
    def global_moran_i(self):
        """Calcula o I de Moran global"""
        if self.weights is None:
            self.create_spatial_weights()
        
        y = self.gdf[self.value_column].values
        moran = Moran(y, self.weights)
        
        return {
            'I': moran.I,
            'p_value': moran.p_sim,
            'z_score': moran.z_sim,
            'expected_i': moran.EI
        }
    
    def local_moran_i(self):
        """Calcula o I de Moran local (LISA)"""
        if self.weights is None:
            self.create_spatial_weights()
        
        y = self.gdf[self.value_column].values
        local_moran = Moran_Local(y, self.weights)
        
        self.gdf['local_i'] = local_moran.Is
        self.gdf['local_p'] = local_moran.p_sim
        self.gdf['local_z'] = local_moran.z_sim
        self.gdf['quadrant'] = local_moran.q
        
        return local_moran
    
    def geary_c(self):
        """Calcula o C de Geary"""
        if self.weights is None:
            self.create_spatial_weights()
        
        y = self.gdf[self.value_column].values
        geary = Geary(y, self.weights)
        
        return {
            'C': geary.C,
            'p_value': geary.p_sim,
            'z_score': geary.z_sim
        }
    
    def getis_ord_g(self):
        """Calcula estatística G de Getis-Ord"""
        if self.weights is None:
            self.create_spatial_weights()
        
        y = self.gdf[self.value_column].values
        g = G(y, self.weights)
        
        return {
            'G': g.G,
            'p_value': g.p_sim,
            'z_score': g.z_sim
        }
    
    def hot_cold_spots(self):
        """Identifica hot spots e cold spots"""
        if self.weights is None:
            self.create_spatial_weights()
        
        y = self.gdf[self.value_column].values
        local_g = G_Local(y, self.weights)
        
        self.gdf['g_local'] = local_g.Zs
        self.gdf['g_p'] = local_g.p_sim
        
        # Classificar hot/cold spots com 95% de confiança
        self.gdf['hot_cold_spot'] = 'Not Significant'
        self.gdf.loc[(self.gdf['g_local'] > 1.96) & (self.gdf['g_p'] < 0.05), 'hot_cold_spot'] = 'Hot Spot'
        self.gdf.loc[(self.gdf['g_local'] < -1.96) & (self.gdf['g_p'] < 0.05), 'hot_cold_spot'] = 'Cold Spot'
        
        return self.gdf
    
    def plot_spatial_autocorrelation(self, figsize=(15, 10)):
        """Plota análise de autocorrelação espacial"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. Mapa de valores
        ax = axes[0, 0]
        self.gdf.plot(column=self.value_column, ax=ax, legend=True,
                     cmap='viridis', edgecolor='black')
        ctx.add_basemap(ax, crs=self.gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
        ax.set_title(f'Mapa de {self.value_column}')
        ax.set_axis_off()
        
        # 2. Histograma
        ax = axes[0, 1]
        self.gdf[self.value_column].hist(ax=ax, bins=30, edgecolor='black')
        ax.set_title(f'Distribuição de {self.value_column}')
        ax.set_xlabel('Valor')
        ax.set_ylabel('Frequência')
        
        # 3. Diagrama de dispersão de Moran
        ax = axes[0, 2]
        moran_result = self.global_moran_i()
        y = self.gdf[self.value_column].values
        plot_moran(moran_result, ax=ax)
        ax.set_title(f"I de Moran = {moran_result['I']:.3f}")
        
        # 4. Cluster Map LISA
        ax = axes[1, 0]
        local_moran = self.local_moran_i()
        lisa_cluster(local_moran, self.gdf, ax=ax, legend=True)
        ctx.add_basemap(ax, crs=self.gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
        ax.set_title('Clusters LISA')
        ax.set_axis_off()
        
        # 5. Hot/Cold Spots
        ax = axes[1, 1]
        self.hot_cold_spots()
        self.gdf.plot(column='hot_cold_spot', ax=ax, categorical=True,
                     legend=True, cmap='coolwarm', edgecolor='black')
        ctx.add_basemap(ax, crs=self.gdf.crs.to_string(), source=ctx.providers.CartoDB.Positron)
        ax.set_title('Hot/Cold Spots (Getis-Ord Gi*)')
        ax.set_axis_off()
        
        # 6. Matriz de pesos espaciais
        ax = axes[1, 2]
        if self.weights is not None:
            # Plotar conexões espaciais
            centroid = self.gdf.geometry.centroid
            for i, neighbors in enumerate(self.weights.neighbors):
                for neighbor in neighbors:
                    x = [centroid.iloc[i].x, centroid.iloc[neighbor].x]
                    y = [centroid.iloc[i].y, centroid.iloc[neighbor].y]
                    ax.plot(x, y, 'k-', alpha=0.3, linewidth=0.5)
            
            self.gdf.plot(ax=ax, color='lightblue', edgecolor='black')
            ax.set_title('Matriz de Pesos Espaciais')
            ax.set_axis_off()
        
        plt.tight_layout()
        return fig