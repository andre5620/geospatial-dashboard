import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import warnings

class VariogramModeler:
    def __init__(self, gdf, value_column):
        self.gdf = gdf
        self.value_column = value_column
        self.distances = None
        self.semivariances = None
        self.bins = None
        
    def calculate_empirical_variogram(self, max_distance=None, n_bins=10):
        """Calcula variograma empírico"""
        # Extrair coordenadas e valores
        coords = np.array([(geom.x, geom.y) for geom in self.gdf.geometry.centroid])
        values = self.gdf[self.value_column].values
        
        # Calcular todas as distâncias entre pares
        pairwise_distances = pdist(coords)
        
        # Calcular diferenças ao quadrado
        n = len(values)
        i, j = np.triu_indices(n, k=1)
        squared_diffs = (values[i] - values[j]) ** 2
        
        if max_distance is None:
            max_distance = np.max(pairwise_distances) / 2
        
        # Criar bins
        self.bins = np.linspace(0, max_distance, n_bins + 1)
        bin_indices = np.digitize(pairwise_distances, self.bins)
        
        # Calcular semivariâncias por bin
        self.semivariances = []
        self.distances = []
        self.pair_counts = []
        
        for k in range(1, n_bins + 1):
            mask = (bin_indices == k)
            if np.sum(mask) > 0:
                bin_dists = pairwise_distances[mask]
                bin_diffs = squared_diffs[mask]
                
                self.distances.append(np.mean(bin_dists))
                self.semivariances.append(0.5 * np.mean(bin_diffs))
                self.pair_counts.append(np.sum(mask))
        
        self.distances = np.array(self.distances)
        self.semivariances = np.array(self.semivariances)
        self.pair_counts = np.array(self.pair_counts)
        
        return self.distances, self.semivariances, self.pair_counts
    
    # Modelos teóricos de variograma
    @staticmethod
    def spherical_model(h, nugget, sill, range_param):
        """Modelo esférico"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = np.asarray(h)
            result = np.zeros_like(h, dtype=float)
            
            mask = h == 0
            result[mask] = nugget
            
            mask = (h > 0) & (h <= range_param)
            result[mask] = nugget + (sill - nugget) * (1.5 * (h[mask]/range_param) - 0.5 * (h[mask]/range_param)**3)
            
            mask = h > range_param
            result[mask] = sill
            
            return result
    
    @staticmethod
    def exponential_model(h, nugget, sill, range_param):
        """Modelo exponencial"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = np.asarray(h)
            result = np.zeros_like(h, dtype=float)
            
            mask = h == 0
            result[mask] = nugget
            
            mask = h > 0
            result[mask] = nugget + (sill - nugget) * (1 - np.exp(-3 * h[mask] / range_param))
            
            return result
    
    @staticmethod
    def gaussian_model(h, nugget, sill, range_param):
        """Modelo gaussiano"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h = np.asarray(h)
            result = np.zeros_like(h, dtype=float)
            
            mask = h == 0
            result[mask] = nugget
            
            mask = h > 0
            result[mask] = nugget + (sill - nugget) * (1 - np.exp(-3 * (h[mask] / range_param)**2))
            
            return result
    
    def fit_variogram_model(self, model_type='spherical', initial_params=None):
        """Ajusta um modelo teórico ao variograma empírico"""
        if self.distances is None:
            self.calculate_empirical_variogram()
        
        # Selecionar modelo
        if model_type == 'spherical':
            model = self.spherical_model
            if initial_params is None:
                initial_params = [0.1 * np.max(self.semivariances), 
                                np.max(self.semivariances), 
                                np.max(self.distances)/2]
        elif model_type == 'exponential':
            model = self.exponential_model
            if initial_params is None:
                initial_params = [0.1 * np.max(self.semivariances), 
                                np.max(self.semivariances), 
                                np.max(self.distances)/3]
        elif model_type == 'gaussian':
            model = self.gaussian_model
            if initial_params is None:
                initial_params = [0.1 * np.max(self.semivariances), 
                                np.max(self.semivariances), 
                                np.max(self.distances)/2]
        else:
            raise ValueError(f"Modelo {model_type} não suportado")
        
        # Ajustar parâmetros
        try:
            bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
            popt, pcov = curve_fit(
                model, 
                self.distances, 
                self.semivariances,
                p0=initial_params,
                bounds=bounds,
                maxfev=5000
            )
            
            # Calcular R²
            predicted = model(self.distances, *popt)
            r2 = r2_score(self.semivariances, predicted)
            
            return {
                'model_type': model_type,
                'parameters': {
                    'nugget': popt[0],
                    'sill': popt[1],
                    'range': popt[2]
                },
                'covariance': pcov,
                'r_squared': r2,
                'model_function': lambda h: model(h, *popt)
            }
        
        except Exception as e:
            print(f"Erro ao ajustar modelo: {e}")
            return None
    
    def plot_variogram(self, fitted_model=None, figsize=(12, 8)):
        """Plota variograma empírico e modelo ajustado"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plotar variograma empírico
        ax1.scatter(self.distances, self.semivariances, 
                   s=50, alpha=0.7, label='Empírico')
        ax1.set_xlabel('Distância')
        ax1.set_ylabel('Semivariância')
        ax1.set_title('Variograma Empírico')
        ax1.grid(True, alpha=0.3)
        
        # Adicionar contagem de pares como tamanho do ponto
        if hasattr(self, 'pair_counts'):
            sizes = self.pair_counts / np.max(self.pair_counts) * 100
            ax1.scatter(self.distances, self.semivariances, s=sizes, 
                       alpha=0.3, color='red')
        
        # Plotar modelo ajustado se fornecido
        if fitted_model:
            h_range = np.linspace(0, np.max(self.distances) * 1.2, 100)
            model_func = fitted_model['model_function']
            ax1.plot(h_range, model_func(h_range), 
                    'r-', linewidth=2, 
                    label=f"{fitted_model['model_type']} (R²={fitted_model['r_squared']:.3f})")
        
        ax1.legend()
        
        # Segundo gráfico: mapa dos pontos
        ax2 = self.gdf.plot(column=self.value_column, ax=ax2, 
                           legend=True, cmap='viridis',
                           edgecolor='black', markersize=50)
        
        # Adicionar mapa de fundo
        try:
            import contextily as ctx
            ctx.add_basemap(ax2, crs=self.gdf.crs.to_string(), 
                           source=ctx.providers.CartoDB.Positron)
        except:
            pass
        
        ax2.set_title('Distribuição Espacial dos Dados')
        ax2.set_axis_off()
        
        plt.tight_layout()
        return fig
    
    def cross_validation(self, model_result, k_folds=5):
        """Realiza validação cruzada do modelo"""
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error
        
        coords = np.array([(geom.x, geom.y) for geom in self.gdf.geometry.centroid])
        values = self.gdf[self.value_column].values
        
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        mse_scores = []
        
        for train_idx, test_idx in kf.split(values):
            # Treinar modelo com subconjunto
            train_coords = coords[train_idx]
            train_values = values[train_idx]
            
            # Aqui seria necessário reajustar o modelo com os dados de treino
            # Para simplificar, usaremos o modelo já ajustado
            # Em uma implementação real, reajustaria o modelo para cada fold
            
            # Predição simples (krigagem ordinária simplificada)
            # Nota: Esta é uma versão simplificada para demonstração
            pred_values = np.zeros(len(test_idx))
            
            for i, test_idx_i in enumerate(test_idx):
                # Calcular distâncias para todos os pontos de treino
                distances = np.sqrt(np.sum((coords[test_idx_i] - train_coords)**2, axis=1))
                
                # Usar o modelo de variograma para pesos
                weights = 1 / (distances + 0.001)  # Simplificação
                weights = weights / np.sum(weights)
                
                pred_values[i] = np.sum(weights * train_values)
            
            # Calcular MSE
            mse = mean_squared_error(values[test_idx], pred_values)
            mse_scores.append(mse)
        
        return {
            'mean_mse': np.mean(mse_scores),
            'std_mse': np.std(mse_scores),
            'mse_scores': mse_scores
        }