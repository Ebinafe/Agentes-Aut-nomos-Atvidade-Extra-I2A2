import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np

class VisualizationTools:
    """
    Ferramentas de visualização
    """
    
    def __init__(self):
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def plot_histogram(self, df: pd.DataFrame, column: str):
        """
        Histograma com estatísticas
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            df[column].hist(ax=ax, bins=30, edgecolor='black', alpha=0.7)
            
            mean_val = df[column].mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Média: {mean_val:.2f}')
            
            median_val = df[column].median()
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                      label=f'Mediana: {median_val:.2f}')
            
            ax.set_title(f'Distribuição de {column}', fontsize=14, fontweight='bold')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequência')
            ax.legend()
            ax.grid(True, alpha=0.3)    
            
            st.pyplot(fig)
            
            st.write(f"""
            **Estatísticas:**
            - Média: {mean_val:.2f}
            - Mediana: {median_val:.2f}
            - Desvio Padrão: {df[column].std():.2f}
            - Mínimo: {df[column].min():.2f}
            - Máximo: {df[column].max():.2f}
            """)
            
            plt.close()
        except Exception as e:
            st.error(f"Erro ao gerar histograma: {str(e)}")
    
    def plot_heatmap(self, df: pd.DataFrame):
        """
        Mapa de calor de correlações
        """
        try:
            numeric_df = df.select_dtypes(include='number')
            
            if len(numeric_df.columns) < 2:
                st.warning("Necessário 2+ colunas numéricas")
                return
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            corr = numeric_df.corr()
            
            sns.heatmap(
                corr, 
                ax=ax, 
                annot=True, 
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8}
            )
            
            ax.set_title('Matriz de Correlação', fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout()
            
            st.pyplot(fig)
            plt.close()
            
            st.write("**Correlações Fortes (|r| > 0.7):**")
            strong_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.7:
                        strong_corr.append({
                            'Variável 1': corr.columns[i],
                            'Variável 2': corr.columns[j],
                            'Correlação': f"{corr.iloc[i, j]:.3f}"
                        })
            
            if strong_corr:
                st.dataframe(pd.DataFrame(strong_corr))
            else:
                st.info("Nenhuma correlação forte")
                
        except Exception as e:
            st.error(f"Erro ao gerar mapa: {str(e)}")
    
    def plot_outliers(self, df: pd.DataFrame, column: str, outliers: pd.DataFrame):
        """
        Visualiza outliers
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            df[column].plot(kind='box', ax=ax1, vert=True)
            ax1.set_title(f'Boxplot - {column}', fontsize=12, fontweight='bold')
            ax1.set_ylabel(column)
            ax1.grid(True, alpha=0.3)
            
            normal_data = df[~df.index.isin(outliers.index)]
            ax2.scatter(normal_data.index, normal_data[column], 
                       alpha=0.5, label='Normal', s=20)
            ax2.scatter(outliers.index, outliers[column], 
                       color='red', alpha=0.7, label='Outliers', s=50)
            ax2.set_title(f'Distribuição com Outliers - {column}', 
                         fontsize=12, fontweight='bold')
            ax2.set_xlabel('Índice')
            ax2.set_ylabel(column)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"Erro ao visualizar outliers: {str(e)}")


    def plot_scatter(self, df: pd.DataFrame, col_x: str, col_y: str):
        """
        Grafico de dispersao entre duas variaveis
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.scatter(df[col_x], df[col_y], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel(col_x, fontsize=12)
            ax.set_ylabel(col_y, fontsize=12)
            ax.set_title(f'Dispersao: {col_x} vs {col_y}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Linha de tendencia
            z = np.polyfit(df[col_x].dropna(), df[col_y].dropna(), 1)
            p = np.poly1d(z)
            ax.plot(df[col_x], p(df[col_x]), "r--", alpha=0.8, linewidth=2, label='Tendencia')
            ax.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Correlacao
            corr = df[[col_x, col_y]].corr().iloc[0, 1]
            st.write(f"**Correlacao:** {corr:.3f}")
            
        except Exception as e:
            st.error(f"Erro ao gerar scatter plot: {str(e)}")

