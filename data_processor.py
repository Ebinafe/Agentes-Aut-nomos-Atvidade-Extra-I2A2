import pandas as pd
import streamlit as st

def load_csv(uploaded_file):
    """
    Carrega CSV otimizado para arquivos grandes
    """
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
        return df
    except UnicodeDecodeError:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1', low_memory=False)
            return df
        except Exception as e:
            st.error(f"Erro ao ler CSV: {e}")
            return None
    except Exception as e:
        st.error(f"Erro ao carregar: {e}")
        return None

def get_basic_stats(df):
    """
    Retorna estatísticas básicas
    """
    stats = {
        "Numero de linhas": int(len(df)),
        "Numero de colunas": int(len(df.columns)),
        "Colunas": [str(col) for col in df.columns],
        "Tipos de dados": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
        "Memoria utilizada": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        "Valores nulos": {str(k): int(v) for k, v in df.isnull().sum().items()},
        "Duplicatas": int(df.duplicated().sum())
    }
    return stats

def prepare_dataframe_info(df):
    """
    Prepara informações para o agente
    """
    info = f"""
    INFORMAÇÕES DO DATASET:
    - Total de linhas: {len(df)}
    - Total de colunas: {len(df.columns)}
    - Colunas: {', '.join(df.columns)}
    
    TIPOS DE DADOS:
    {df.dtypes.to_string()}
    
    ESTATÍSTICAS:
    {df.describe(include='all').to_string()}
    
    PRIMEIRAS LINHAS:
    {df.head(3).to_string()}
    """
    return info