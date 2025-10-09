<<<<<<< HEAD
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
SRC_PATH = ROOT / "src"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    

import streamlit as st
import pandas as pd
import re 
import unidecode


from src.utils.data_processor import get_basic_stats
from src.agent.eda_agent import EDAAgent
from src.tools.statistical_tools import StatisticalTools
from src.tools.visualization_tools import VisualizationTools
from src.tools.outlier_detection import OutlierDetector
from src.tools.pattern_recognition import PatternRecognizer
from src.agent.eda_agent import EDAAgent


def clean_column_names(df):
    new_columns = {}
    for col in df.columns:
        clean_name = unidecode.unidecode(str(col))
        clean_name = clean_name.lower()
        clean_name = clean_name.replace(' ', '_')
        clean_name = re.sub(r'[^a-z0-9_]', '', clean_name)
        new_columns[col] = clean_name
    return df.rename(columns=new_columns)

def load_csv_with_encoding_fallback(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
        st.success("[OK] Arquivo CSV carregado com UTF-8.")
        return df
    except UnicodeDecodeError:
        try:
            st.warning("[AVISO] Tentando Latin-1...")
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1', low_memory=False) 
            st.warning("[OK] Arquivo carregado com Latin-1.")
            return df
        except Exception as e:
            st.error(f"[ERRO] Nao foi possivel ler o CSV: {e}")
            st.stop()
    except Exception as e:
        st.error(f"[ERRO] Erro inesperado: {e}")
        st.stop()
    return None

st.set_page_config(
    page_title="Agente IA - Analise CSV",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

if "df" not in st.session_state:
    st.session_state.df = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown('<h1 class="main-header">Agente Autonomo de Analise CSV</h1>', unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader(
    "Envie seu CSV", 
    type=['csv'],
    help="Apenas arquivos CSV sao suportados."
)

if uploaded_file is not None and st.session_state.df is None:
    df = load_csv_with_encoding_fallback(uploaded_file)
    
    if df is not None:
        with st.spinner("Inicializando Agente..."):
            df = clean_column_names(df) 
            st.session_state.df = df
            st.session_state.agent = EDAAgent(st.session_state.df)
            st.info(f"[OK] Agente inicializado: {len(df)} linhas, {len(df.columns)} colunas.")
        st.rerun()

if st.session_state.df is None:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.info("""
        ### Bem-vindo ao Agente de Analise de Dados!
        
        **Como usar:**
        1. Faca upload de um arquivo CSV acima
        2. Explore as 4 abas disponiveis
        3. Converse com o agente ou visualize analises
        
        **Abas disponiveis:**
        - **Chat:** Converse diretamente com o agente
        - **Analises Rapidas:** Tabelas estatisticas completas
        - **Visualizacoes:** Graficos e mapas de calor
        - **Conclusoes:** Insights consolidados do agente
        """)
else:
    df = st.session_state.df
    agent = st.session_state.agent
    
    # INFORMAÇÕES DO TOPO (fora das tabs)
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Linhas", f"{len(df):,}")
    with col2:
        st.metric("Colunas", len(df.columns))
    with col3:
        st.metric("Memoria", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        if agent:
            try:
                model_info = agent.get_current_model_info()
                st.metric("Modelo", model_info['name'])
            except:
                pass
    
    st.markdown("---")
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat",
        "📊 Analises Rapidas", 
        "📈 Visualizacoes",
        "🎯 Conclusoes"
    ])
    
    # ==================== TAB 1: CHAT PURO ====================
    with tab1:
        st.subheader("Chat com o Agente")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Limpar Chat"):
                st.session_state.chat_history = []
                if agent:
                    agent.clear_history()
                st.success("Chat limpo!")
                st.rerun()
        
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["query"])
            with st.chat_message("assistant"):
                st.write(chat["response"])
        
        user_query = st.chat_input("Digite sua pergunta sobre os dados...")
        
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            
            with st.chat_message("assistant"):
                with st.spinner("Analisando..."):
                    response = agent.ask(user_query)
                    st.write(response)
            
            st.session_state.chat_history.append({
                "query": user_query,
                "response": response
            })
            st.rerun()
    
    # ==================== TAB 2: ANÁLISES PURAS ====================
    with tab2:
        st.subheader("Analises Estatisticas Completas")
        
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        
        if numeric_cols:
            # TABELA 1: Estatísticas Descritivas
            st.write("### Estatisticas Descritivas")
            stats_data = []
            for col in numeric_cols:
                stats_data.append({
                    'Coluna': col,
                    'Minimo': f"{df[col].min():.4f}",
                    'Maximo': f"{df[col].max():.4f}",
                    'Media': f"{df[col].mean():.4f}",
                    'Mediana': f"{df[col].median():.4f}",
                    'Desvio Padrao': f"{df[col].std():.4f}",
                    'Variancia': f"{df[col].var():.4f}"
                })
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width='stretch')
            
            st.markdown("---")
            
            # TABELA 2: Tipos de Dados
            st.write("### Tipos de Dados por Coluna")
            types_data = []
            for col in df.columns:
                types_data.append({
                    'Coluna': col,
                    'Tipo': str(df[col].dtype),
                    'Nulos': int(df[col].isnull().sum()),
                    'Unicos': int(df[col].nunique())
                })
            
            st.dataframe(pd.DataFrame(types_data), use_container_width='stretch')
            
            st.markdown("---")
            
            # BOTÕES ADICIONAIS
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Estatisticas Pandas Completas"):
                    st.dataframe(df.describe(), use_container_width='stretch')
            
            with col2:
                if st.button("Matriz de Correlacao"):
                    stat_tools = StatisticalTools()
                    corr = stat_tools.correlation(df)
                    st.dataframe(corr, use_container_width='stretch')
        else:
            st.warning("Nenhuma coluna numerica encontrada")
    
    # ==================== TAB 3: VISUALIZAÇÕES PURAS ====================
    with tab3:
        st.subheader("Visualizacoes e Graficos")
        
        vis = VisualizationTools()
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        
        if numeric_cols:
            # HISTOGRAMA
            st.write("### Histograma")
            col1, col2 = st.columns([3, 1])
            with col1:
                hist_col = st.selectbox("Coluna para histograma:", numeric_cols, key="hist")
            with col2:
                if st.button("Gerar Histograma"):
                    vis.plot_histogram(df, hist_col)
            
            st.markdown("---")
            
            # MAPA DE CALOR
            st.write("### Mapa de Calor (Correlacoes)")
            if st.button("Gerar Mapa de Calor"):
                vis.plot_heatmap(df)
            
            st.markdown("---")
            
            # SCATTER PLOT
            st.write("### Grafico de Dispersao")
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                scatter_x = st.selectbox("Eixo X:", numeric_cols, key="scatter_x")
            with col2:
                scatter_y = st.selectbox("Eixo Y:", numeric_cols, key="scatter_y")
            with col3:
                if st.button("Gerar Scatter"):
                    vis.plot_scatter(df, scatter_x, scatter_y)
            
            st.markdown("---")
            
            # OUTLIERS
            st.write("### Deteccao de Outliers")
            outlier_detector = OutlierDetector()
            col1, col2 = st.columns([3, 1])
            with col1:
                outlier_col = st.selectbox("Coluna para outliers:", numeric_cols, key="outlier")
            with col2:
                if st.button("Detectar Outliers"):
                    outliers = outlier_detector.detect(df, outlier_col)
                    st.write(f"**Outliers: {len(outliers)}**")
                    if len(outliers) > 0:
                        st.dataframe(outliers.head(20), use_container_width='stretch')
                        vis.plot_outliers(df, outlier_col, outliers)
        else:
            st.warning("Nenhuma coluna numerica")
        
        st.markdown("---")
        
        # PADRÕES CATEGÓRICOS
        st.write("### Padroes em Variaveis Categoricas")
        pattern_recognizer = PatternRecognizer()
        
        if st.button("Encontrar Padroes"):
            patterns = pattern_recognizer.find_patterns(df)
            if patterns:
                for col, values in patterns.items():
                    with st.expander(f"Coluna: {col}"):
                        st.write(values)
            else:
                st.info("Sem variaveis categoricas")
    
    # ==================== TAB 4: CONCLUSÕES PURAS ====================
    with tab4:
        st.subheader("Conclusoes Consolidadas do Agente")
        
        if len(st.session_state.chat_history) == 0:
            st.info("""
            ### Como gerar conclusoes:
            
            1. Va para a aba **Chat**
            2. Faca pelo menos 3-4 perguntas sobre os dados
            3. Volte aqui e clique em **Gerar Conclusoes**
            
            O agente analisara todo o historico de perguntas e respostas
            para gerar insights consolidados sobre o dataset.
            """)
        else:
            st.success(f"Historico: {len(st.session_state.chat_history)} perguntas realizadas")
            
            if st.button("Gerar Conclusoes Consolidadas", type="primary"):
                with st.spinner("Analisando todo o historico e gerando conclusoes..."):
                    conclusions = agent.get_conclusions()
                    
                    st.markdown("### Conclusoes do Agente:")
                    st.write(conclusions)
                    
                    st.download_button(
                        label="Baixar Conclusoes",
                        data=conclusions,
                        file_name="conclusoes_analise.txt",
                        mime="text/plain"
                    )

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Agente EDA | Powered by OpenRouter + LangChain</p>
</div>
=======
import sys
import os

if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

import streamlit as st
import pandas as pd
import re 
import unidecode
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.utils.data_processor import get_basic_stats
from src.agent.eda_agent import EDAAgent
from src.tools.statistical_tools import StatisticalTools
from src.tools.visualization_tools import VisualizationTools
from src.tools.outlier_detection import OutlierDetector
from src.tools.pattern_recognition import PatternRecognizer
from src.agent.eda_agent import EDAAgent


def clean_column_names(df):
    new_columns = {}
    for col in df.columns:
        clean_name = unidecode.unidecode(str(col))
        clean_name = clean_name.lower()
        clean_name = clean_name.replace(' ', '_')
        clean_name = re.sub(r'[^a-z0-9_]', '', clean_name)
        new_columns[col] = clean_name
    return df.rename(columns=new_columns)

def load_csv_with_encoding_fallback(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8', low_memory=False)
        st.success("[OK] Arquivo CSV carregado com UTF-8.")
        return df
    except UnicodeDecodeError:
        try:
            st.warning("[AVISO] Tentando Latin-1...")
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1', low_memory=False) 
            st.warning("[OK] Arquivo carregado com Latin-1.")
            return df
        except Exception as e:
            st.error(f"[ERRO] Nao foi possivel ler o CSV: {e}")
            st.stop()
    except Exception as e:
        st.error(f"[ERRO] Erro inesperado: {e}")
        st.stop()
    return None

st.set_page_config(
    page_title="Agente IA - Analise CSV",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

if "df" not in st.session_state:
    st.session_state.df = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.markdown('<h1 class="main-header">Agente Autonomo de Analise CSV</h1>', unsafe_allow_html=True)
st.markdown("---")

uploaded_file = st.file_uploader(
    "Envie seu CSV", 
    type=['csv'],
    help="Apenas arquivos CSV sao suportados."
)

if uploaded_file is not None and st.session_state.df is None:
    df = load_csv_with_encoding_fallback(uploaded_file)
    
    if df is not None:
        with st.spinner("Inicializando Agente..."):
            df = clean_column_names(df) 
            st.session_state.df = df
            st.session_state.agent = EDAAgent(st.session_state.df)
            st.info(f"[OK] Agente inicializado: {len(df)} linhas, {len(df.columns)} colunas.")
        st.rerun()

if st.session_state.df is None:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.info("""
        ### Bem-vindo ao Agente de Analise de Dados!
        
        **Como usar:**
        1. Faca upload de um arquivo CSV acima
        2. Explore as 4 abas disponiveis
        3. Converse com o agente ou visualize analises
        
        **Abas disponiveis:**
        - **Chat:** Converse diretamente com o agente
        - **Analises Rapidas:** Tabelas estatisticas completas
        - **Visualizacoes:** Graficos e mapas de calor
        - **Conclusoes:** Insights consolidados do agente
        """)
else:
    df = st.session_state.df
    agent = st.session_state.agent
    
    # INFORMAÇÕES DO TOPO (fora das tabs)
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Linhas", f"{len(df):,}")
    with col2:
        st.metric("Colunas", len(df.columns))
    with col3:
        st.metric("Memoria", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        if agent:
            try:
                model_info = agent.get_current_model_info()
                st.metric("Modelo", model_info['name'])
            except:
                pass
    
    st.markdown("---")
    
    # TABS
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat",
        "📊 Analises Rapidas", 
        "📈 Visualizacoes",
        "🎯 Conclusoes"
    ])
    
    # ==================== TAB 1: CHAT PURO ====================
    with tab1:
        st.subheader("Chat com o Agente")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Limpar Chat"):
                st.session_state.chat_history = []
                if agent:
                    agent.clear_history()
                st.success("Chat limpo!")
                st.rerun()
        
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["query"])
            with st.chat_message("assistant"):
                st.write(chat["response"])
        
        user_query = st.chat_input("Digite sua pergunta sobre os dados...")
        
        if user_query:
            with st.chat_message("user"):
                st.write(user_query)
            
            with st.chat_message("assistant"):
                with st.spinner("Analisando..."):
                    response = agent.ask(user_query)
                    st.write(response)
            
            st.session_state.chat_history.append({
                "query": user_query,
                "response": response
            })
            st.rerun()
    
    # ==================== TAB 2: ANÁLISES PURAS ====================
    with tab2:
        st.subheader("Analises Estatisticas Completas")
        
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        
        if numeric_cols:
            # TABELA 1: Estatísticas Descritivas
            st.write("### Estatisticas Descritivas")
            stats_data = []
            for col in numeric_cols:
                stats_data.append({
                    'Coluna': col,
                    'Minimo': f"{df[col].min():.4f}",
                    'Maximo': f"{df[col].max():.4f}",
                    'Media': f"{df[col].mean():.4f}",
                    'Mediana': f"{df[col].median():.4f}",
                    'Desvio Padrao': f"{df[col].std():.4f}",
                    'Variancia': f"{df[col].var():.4f}"
                })
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width='stretch')
            
            st.markdown("---")
            
            # TABELA 2: Tipos de Dados
            st.write("### Tipos de Dados por Coluna")
            types_data = []
            for col in df.columns:
                types_data.append({
                    'Coluna': col,
                    'Tipo': str(df[col].dtype),
                    'Nulos': int(df[col].isnull().sum()),
                    'Unicos': int(df[col].nunique())
                })
            
            st.dataframe(pd.DataFrame(types_data), use_container_width='stretch')
            
            st.markdown("---")
            
            # BOTÕES ADICIONAIS
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Estatisticas Pandas Completas"):
                    st.dataframe(df.describe(), use_container_width='stretch')
            
            with col2:
                if st.button("Matriz de Correlacao"):
                    stat_tools = StatisticalTools()
                    corr = stat_tools.correlation(df)
                    st.dataframe(corr, use_container_width='stretch')
        else:
            st.warning("Nenhuma coluna numerica encontrada")
    
    # ==================== TAB 3: VISUALIZAÇÕES PURAS ====================
    with tab3:
        st.subheader("Visualizacoes e Graficos")
        
        vis = VisualizationTools()
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        
        if numeric_cols:
            # HISTOGRAMA
            st.write("### Histograma")
            col1, col2 = st.columns([3, 1])
            with col1:
                hist_col = st.selectbox("Coluna para histograma:", numeric_cols, key="hist")
            with col2:
                if st.button("Gerar Histograma"):
                    vis.plot_histogram(df, hist_col)
            
            st.markdown("---")
            
            # MAPA DE CALOR
            st.write("### Mapa de Calor (Correlacoes)")
            if st.button("Gerar Mapa de Calor"):
                vis.plot_heatmap(df)
            
            st.markdown("---")
            
            # SCATTER PLOT
            st.write("### Grafico de Dispersao")
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                scatter_x = st.selectbox("Eixo X:", numeric_cols, key="scatter_x")
            with col2:
                scatter_y = st.selectbox("Eixo Y:", numeric_cols, key="scatter_y")
            with col3:
                if st.button("Gerar Scatter"):
                    vis.plot_scatter(df, scatter_x, scatter_y)
            
            st.markdown("---")
            
            # OUTLIERS
            st.write("### Deteccao de Outliers")
            outlier_detector = OutlierDetector()
            col1, col2 = st.columns([3, 1])
            with col1:
                outlier_col = st.selectbox("Coluna para outliers:", numeric_cols, key="outlier")
            with col2:
                if st.button("Detectar Outliers"):
                    outliers = outlier_detector.detect(df, outlier_col)
                    st.write(f"**Outliers: {len(outliers)}**")
                    if len(outliers) > 0:
                        st.dataframe(outliers.head(20), use_container_width='stretch')
                        vis.plot_outliers(df, outlier_col, outliers)
        else:
            st.warning("Nenhuma coluna numerica")
        
        st.markdown("---")
        
        # PADRÕES CATEGÓRICOS
        st.write("### Padroes em Variaveis Categoricas")
        pattern_recognizer = PatternRecognizer()
        
        if st.button("Encontrar Padroes"):
            patterns = pattern_recognizer.find_patterns(df)
            if patterns:
                for col, values in patterns.items():
                    with st.expander(f"Coluna: {col}"):
                        st.write(values)
            else:
                st.info("Sem variaveis categoricas")
    
    # ==================== TAB 4: CONCLUSÕES PURAS ====================
    with tab4:
        st.subheader("Conclusoes Consolidadas do Agente")
        
        if len(st.session_state.chat_history) == 0:
            st.info("""
            ### Como gerar conclusoes:
            
            1. Va para a aba **Chat**
            2. Faca pelo menos 3-4 perguntas sobre os dados
            3. Volte aqui e clique em **Gerar Conclusoes**
            
            O agente analisara todo o historico de perguntas e respostas
            para gerar insights consolidados sobre o dataset.
            """)
        else:
            st.success(f"Historico: {len(st.session_state.chat_history)} perguntas realizadas")
            
            if st.button("Gerar Conclusoes Consolidadas", type="primary"):
                with st.spinner("Analisando todo o historico e gerando conclusoes..."):
                    conclusions = agent.get_conclusions()
                    
                    st.markdown("### Conclusoes do Agente:")
                    st.write(conclusions)
                    
                    st.download_button(
                        label="Baixar Conclusoes",
                        data=conclusions,
                        file_name="conclusoes_analise.txt",
                        mime="text/plain"
                    )

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Agente EDA | Powered by OpenRouter + LangChain</p>
</div>
>>>>>>> d4ead596466e6316ed201c1b2862c7a17a1c2125
""", unsafe_allow_html=True)