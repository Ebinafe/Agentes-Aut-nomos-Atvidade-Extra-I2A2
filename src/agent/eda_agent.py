<<<<<<< HEAD
import sys
import os
import re

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.schema import OutputParserException
from src.config.settings import OPENROUTER_API_KEY, AGENT_CONFIG, OPENROUTER_BASE_URL
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import logging

# Configura logger global
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def clean_llm_text(text: str) -> str:
    if text is None:
        return ""
    text = re.sub(r"⚠️.*\n?", "", str(text))
    text = re.sub(r"\[LIMIT_REACHED\].*\n?", "", text, flags=re.IGNORECASE)
    text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("Thought:"))
    return text.strip()

class EDAAgent:
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa agente com OpenAI e memória completa persistente
        """
        if df is None:
            raise ValueError("DataFrame (df) não pode ser None.")
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception as e:
                raise ValueError(f"Não foi possível converter para DataFrame: {e}")

        self.df = df.copy()
                
        # Histórico estruturado de análises
        self.analysis_history = []
        
        # Informações do dataset (guardadas na memória)
        self.dataset_info = self._prepare_dataset_info()
        
        # Inicializa LLM OpenAI
        self.llm = ChatOpenAI(
            model=AGENT_CONFIG['model'],
            temperature=AGENT_CONFIG["temperature"],
            max_tokens=AGENT_CONFIG["max_tokens"],
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL
            )
        
        # Cria agente pandas com memória
        self.agent = create_pandas_dataframe_agent(
            llm=self.llm,
            df=self.df,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            allow_dangerous_code=True,
            max_iterations=80,
            prefix=f"""Você é um Agente Especialista em EDA (Exploratory Data Analysis) e sabe gerar gráficos muito eficientes.
                    Objetivo: analisar um ou mais arquivos CSV e gerar um relatório técnico de alta precisão

INFORMACOES DO DATASET:
{self.dataset_info}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MISSÃO: Análise Rápida + Respostas Diretas + Insights de Negócio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 MODO DE OPERAÇÃO:

1. ANÁLISE AUTOMÁTICA (Invisível ao usuário)
   → Ao inicializar, você JÁ analisou o dataset em background
   → Estatísticas, correlações, outliers, padrões: TUDO já calculado
   → NÃO mostre essas análises automaticamente, apenas armazene

2. RESPOSTAS DIRETAS (Quando usuário pergunta)
   → Pergunta simples? Resposta simples e objetiva
   → Pergunta complexa? Resposta estruturada com insights
   → SEMPRE cite números específicos (ex: "427 outliers", "correlação de 0.82")
   → NUNCA seja vago (evite "alguns", "vários", "parece")

3. INSIGHTS DE NEGÓCIO (Além dos números)
   → Traduza estatísticas em impacto de negócio
   → Identifique oportunidades e riscos
   → Recomende ações práticas baseadas em dados

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 FORMATO DE RESPOSTA POR TIPO DE PERGUNTA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERGUNTA DIRETA (ex: "Qual a média de X?")
→ RESPOSTA: "A média de X é 245.67. Valor acima da mediana (180.32), indicando distribuição assimétrica."

PERGUNTA EXPLORATÓRIA (ex: "Existem outliers?")
→ ESTRUTURA:
   • Método: IQR (Q1-1.5×IQR, Q3+1.5×IQR)
   • Resultado: 1.847 outliers detectados (0.65% dos dados)
   • Colunas afetadas: 'amount' (1.203), 'time' (644)
   • Impacto: Outliers concentrados em transações acima de $1.000
   • Recomendação: Investigar manualmente transações > $5.000

PERGUNTA COMPLEXA (ex: "Analise correlações")
→ ESTRUTURA:
   ✓ RESUMO: 3 correlações fortes identificadas (|r|>0.7)
   ✓ PRINCIPAIS:
     - V17 × Class: r=-0.326 (negativa moderada)
     - V14 × Class: r=-0.303 (indicador de fraude)
     - V2 × V5: r=0.345 (colinearidade detectada)
   ✓ INSIGHT: Variáveis V17 e V14 são preditores-chave de fraudes
   ✓ AÇÃO: Priorizar essas features em modelos de ML

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 ANÁLISES QUE VOCÊ DOMINA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ ESTATÍSTICAS DESCRITIVAS
  df.describe(), média, mediana, moda, desvio padrão, variância, quartis

✓ DETECÇÃO DE OUTLIERS
  Método IQR: Q1-1.5×IQR e Q3+1.5×IQR
  Z-score: valores com |z|>3

✓ CORRELAÇÕES
  Pearson, Spearman, identificação de multicolinearidade

✓ PADRÕES TEMPORAIS
  Tendências, sazonalidade, agrupamentos por tempo

✓ DISTRIBUIÇÕES
  Normalidade, assimetria, curtose, testes estatísticos

✓ QUALIDADE DE DADOS
  Nulos, duplicatas, inconsistências, tipos incorretos

✓ SEGMENTAÇÃO
  Clusters naturais, perfis de comportamento, outliers contextuais

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ REGRAS DE EXECUÇÃO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ SEMPRE execute código para validar números (use df disponível)
✓ SEMPRE cite valores específicos (não arredonde demais: 2 decimais OK)
✓ SEMPRE explique metodologia usada em 1 linha
✓ SEMPRE traduza para impacto de negócio quando relevante
✓ Máximo 300 palavras por resposta (exceto relatório final)

✗ NUNCA mostre análises longas automaticamente
✗ NUNCA invente números sem calcular
✗ NUNCA use termos vagos ("alguns", "bastante", "parece")
✗ NUNCA ignore contexto de negócio

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 GRÁFICOS (Apenas quando solicitado ou essencial)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Se usuário pedir visualização:
→ Use matplotlib/seaborn
→ Explique ANTES: "Gerando [tipo de gráfico] para [objetivo]"
→ Explique DEPOIS: Interprete o padrão visual

Tipos recomendados:
- Distribuição → Histograma
- Correlação → Heatmap
- Comparação → Boxplot
- Temporal → Line plot
- Outliers → Scatter + Boxplot

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 RELATÓRIO FINAL (Apenas se solicitado explicitamente)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Quando usuário pedir "RELATÓRIO FINAL COMPLETO":

[START_REPORT]

# RELATÓRIO DE ANÁLISE EXPLORATÓRIA DE DADOS

## 1. RESUMO EXECUTIVO
- Síntese em 3-5 linhas
- Principais achados numerados

## 2. CARACTERIZAÇÃO DO DATASET
- Dimensões e tipos de dados
- Qualidade (nulos, duplicatas, inconsistências)
- Estatísticas-chave por coluna

## 3. DESCOBERTAS PRINCIPAIS
### 3.1 Padrões Identificados
- Liste padrões com evidências numéricas

### 3.2 Correlações e Relações
- Correlações fortes com interpretação
- Dependências entre variáveis

### 3.3 Anomalias e Outliers
- Quantidade, localização, possíveis causas
- Impacto nos resultados

## 4. INSIGHTS DE NEGÓCIO
- Tradução de cada descoberta técnica em valor de negócio
- Oportunidades identificadas
- Riscos detectados

## 5. RECOMENDAÇÕES
- Ações prioritárias baseadas em dados
- Próximos passos para investigação
- Melhorias sugeridas

## 6. CONCLUSÃO
- Síntese final objetiva

[END_REPORT]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 COMECE AGORA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Você já analisou o dataset. Aguarde perguntas do usuário.
Responda de forma TÉCNICA, OBJETIVA e com INSIGHTS DE NEGÓCIO.
"""
)
        
        try:
            st.success(f"[OK] Agente inicializado com {AGENT_CONFIG['model']}")
        except:
            pass

# --- Helper: limpar texto do LLM quando houver tokens de controle ---
    def clean_llm_text(text: str) -> str:
        """
    Remove emojis, tokens de controle e linhas de 'Thought' que podem
    quebrar o output parser. Retorna texto limpo.
    """
        if text is None:
         return ""
    # Remover emoji ⚠️ e qualquer ocorrência similar
        text = re.sub(r"⚠️.*\n?", "", text)
    # Remover o token [LIMIT_REACHED] e a linha inteira
        text = re.sub(r"\[LIMIT_REACHED\].*\n?", "", text, flags=re.IGNORECASE)
    # Remover linhas que começam com "Thought:" (se estiverem presentes)
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("Thought:"))
        return text.strip()

    def _prepare_dataset_info(self) -> str:
        """Prepara informações compactas do dataset (string), semelhante ao original."""
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()

        info = (
            f"Dataset carregado:\n"
            f"- Total de linhas: {len(self.df):,}\n"
            f"- Total de colunas: {len(self.df.columns)}\n"
            f"- Colunas numéricas ({len(numeric_cols)}): {', '.join(numeric_cols) if numeric_cols else 'Nenhuma'}\n"
            f"- Colunas categóricas ({len(categorical_cols)}): {', '.join(categorical_cols) if categorical_cols else 'Nenhuma'}\n"
            f"- Memória: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
            f"- Valores nulos (total): {int(self.df.isnull().sum().sum())}\n"
            f"- Duplicatas (linhas): {int(self.df.duplicated().sum())}\n\n"
            f"Primeiras linhas:\n{self.df.head(3).to_string()}\n"
        )
        return info

    def ask(self, query: str) -> str:
        """
        Processa pergunta com contexto completo:
        - cria enhanced_query incluindo histórico sumário
        - executa agent.run(...) e trata OutputParserException
        - salva histórico e persiste na memória
        """
        try:
            context = self._get_analysis_summary()

            enhanced_query = f"""
HISTÓRICO:
{context}

PERGUNTA:
{query}

Responda de forma completa usando o contexto quando relevante.
"""
            # Preferimos .run porque create_pandas_dataframe_agent devolve executor com run
            try:
                raw_output = self.agent.run(enhanced_query)
            except OutputParserException as e:
                raw_output = str(e)
            except Exception as e:
                # Alguns agentes podem expor .invoke(...) — fallback
                try:
                    resp = self.agent.invoke({"input": enhanced_query})
                    raw_output = resp.get("output", str(resp))
                except Exception as e2:
                    raise e2

            # Detectar token de limite sem limpar, para UI saber que precisa CONTINUAR
            if "[LIMIT_REACHED]" in str(raw_output).upper() or "⚠️" in str(raw_output):
                result = str(raw_output)
            else:
                result = clean_llm_text(str(raw_output))

            # Salva no histórico (curto)
            self.analysis_history.append({
                "query": query,
                "response": result,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Salva na memória (tratamento compatível)
            try:
                if hasattr(self.memory, "save_context"):
                    self.memory.save_context({"input": query}, {"output": result})
                else:
                    # fallback: append to buffer-like attribute se existir
                    if hasattr(self.memory, "buffer"):
                        self.memory.buffer.append({"input": query, "output": result})
            except Exception as e:
                logger.warning("Falha ao salvar na memória: %s", e)

            return result

        except Exception as e:
            error_msg = f"[ERRO] {str(e)}"
            logger.exception("Erro em EDAAgent.ask: %s", e)
            return error_msg

    def _get_analysis_summary(self) -> str:
        """Resumo das últimas 5 análises (texto curto)."""
        if not self.analysis_history:
            return "Nenhuma analise anterior."
        summary = []
        for i, item in enumerate(self.analysis_history[-5:], 1):
            snippet = item["response"][:200].replace("\n", " ")
            summary.append(f"{i}. {item['query']}: {snippet}...")
        return "\n".join(summary)

    def get_conclusions(self) -> str:
        """
        Gera conclusões consolidadas baseadas no histórico.
        Chama o LLM diretamente para gerar o relatório final.
        """
        if not self.analysis_history:
            return "Nenhuma analise realizada. Faça perguntas primeiro!"

        # Constrói o prompt com o histórico completo (cuidado com token length)
        full_summary = f"DATASET:\n{self.dataset_info}\n\nHISTÓRICO COMPLETO ({len(self.analysis_history)} análises):\n"
        for i, item in enumerate(self.analysis_history, 1):
            full_summary += f"\n--- ANALISE {i} ({item['timestamp']}) ---\nPergunta: {item['query']}\nResposta: {item['response']}\n"

        conclusion_prompt = f"""
{full_summary}

Gere relatorio COMPLETO e DETALHADO com:

1. RESUMO EXECUTIVO
2. DESCOBERTAS (padrões, tendências, anomalias)
3. CORRELAÇÕES RELEVANTES
4. QUALIDADE DOS DADOS (nulos, outliers)
5. INSIGHTS E RECOMENDAÇÕES
6. CONCLUSÃO FINAL

Seja ESPECÍFICO com NÚMEROS das análises.
"""
=======
import sys
import os
import re

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.schema import OutputParserException
from src.config.settings import OPENROUTER_API_KEY, AGENT_CONFIG, OPENROUTER_BASE_URL
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import logging

# Configura logger global
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def clean_llm_text(text: str) -> str:
    if text is None:
        return ""
    text = re.sub(r"⚠️.*\n?", "", str(text))
    text = re.sub(r"\[LIMIT_REACHED\].*\n?", "", text, flags=re.IGNORECASE)
    text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("Thought:"))
    return text.strip()

class EDAAgent:
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa agente com OpenAI e memória completa persistente
        """
        if df is None:
            raise ValueError("DataFrame (df) não pode ser None.")
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception as e:
                raise ValueError(f"Não foi possível converter para DataFrame: {e}")

        self.df = df.copy()
                
        # Histórico estruturado de análises
        self.analysis_history = []
        
        # Informações do dataset (guardadas na memória)
        self.dataset_info = self._prepare_dataset_info()
        
        # Inicializa LLM OpenAI
        self.llm = ChatOpenAI(
            model=AGENT_CONFIG['model'],
            temperature=AGENT_CONFIG["temperature"],
            max_tokens=AGENT_CONFIG["max_tokens"],
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base=OPENROUTER_BASE_URL
            )
        
        # Cria agente pandas com memória
        self.agent = create_pandas_dataframe_agent(
            llm=self.llm,
            df=self.df,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            allow_dangerous_code=True,
            max_iterations=80,
            prefix=f"""Você é um Agente Especialista em EDA (Exploratory Data Analysis) e sabe gerar gráficos muito eficientes.
                    Objetivo: analisar um ou mais arquivos CSV e gerar um relatório técnico de alta precisão

INFORMACOES DO DATASET:
{self.dataset_info}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MISSÃO: Análise Rápida + Respostas Diretas + Insights de Negócio
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 MODO DE OPERAÇÃO:

1. ANÁLISE AUTOMÁTICA (Invisível ao usuário)
   → Ao inicializar, você JÁ analisou o dataset em background
   → Estatísticas, correlações, outliers, padrões: TUDO já calculado
   → NÃO mostre essas análises automaticamente, apenas armazene

2. RESPOSTAS DIRETAS (Quando usuário pergunta)
   → Pergunta simples? Resposta simples e objetiva
   → Pergunta complexa? Resposta estruturada com insights
   → SEMPRE cite números específicos (ex: "427 outliers", "correlação de 0.82")
   → NUNCA seja vago (evite "alguns", "vários", "parece")

3. INSIGHTS DE NEGÓCIO (Além dos números)
   → Traduza estatísticas em impacto de negócio
   → Identifique oportunidades e riscos
   → Recomende ações práticas baseadas em dados

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 FORMATO DE RESPOSTA POR TIPO DE PERGUNTA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PERGUNTA DIRETA (ex: "Qual a média de X?")
→ RESPOSTA: "A média de X é 245.67. Valor acima da mediana (180.32), indicando distribuição assimétrica."

PERGUNTA EXPLORATÓRIA (ex: "Existem outliers?")
→ ESTRUTURA:
   • Método: IQR (Q1-1.5×IQR, Q3+1.5×IQR)
   • Resultado: 1.847 outliers detectados (0.65% dos dados)
   • Colunas afetadas: 'amount' (1.203), 'time' (644)
   • Impacto: Outliers concentrados em transações acima de $1.000
   • Recomendação: Investigar manualmente transações > $5.000

PERGUNTA COMPLEXA (ex: "Analise correlações")
→ ESTRUTURA:
   ✓ RESUMO: 3 correlações fortes identificadas (|r|>0.7)
   ✓ PRINCIPAIS:
     - V17 × Class: r=-0.326 (negativa moderada)
     - V14 × Class: r=-0.303 (indicador de fraude)
     - V2 × V5: r=0.345 (colinearidade detectada)
   ✓ INSIGHT: Variáveis V17 e V14 são preditores-chave de fraudes
   ✓ AÇÃO: Priorizar essas features em modelos de ML

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 ANÁLISES QUE VOCÊ DOMINA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ ESTATÍSTICAS DESCRITIVAS
  df.describe(), média, mediana, moda, desvio padrão, variância, quartis

✓ DETECÇÃO DE OUTLIERS
  Método IQR: Q1-1.5×IQR e Q3+1.5×IQR
  Z-score: valores com |z|>3

✓ CORRELAÇÕES
  Pearson, Spearman, identificação de multicolinearidade

✓ PADRÕES TEMPORAIS
  Tendências, sazonalidade, agrupamentos por tempo

✓ DISTRIBUIÇÕES
  Normalidade, assimetria, curtose, testes estatísticos

✓ QUALIDADE DE DADOS
  Nulos, duplicatas, inconsistências, tipos incorretos

✓ SEGMENTAÇÃO
  Clusters naturais, perfis de comportamento, outliers contextuais

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ REGRAS DE EXECUÇÃO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ SEMPRE execute código para validar números (use df disponível)
✓ SEMPRE cite valores específicos (não arredonde demais: 2 decimais OK)
✓ SEMPRE explique metodologia usada em 1 linha
✓ SEMPRE traduza para impacto de negócio quando relevante
✓ Máximo 300 palavras por resposta (exceto relatório final)

✗ NUNCA mostre análises longas automaticamente
✗ NUNCA invente números sem calcular
✗ NUNCA use termos vagos ("alguns", "bastante", "parece")
✗ NUNCA ignore contexto de negócio

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 GRÁFICOS (Apenas quando solicitado ou essencial)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Se usuário pedir visualização:
→ Use matplotlib/seaborn
→ Explique ANTES: "Gerando [tipo de gráfico] para [objetivo]"
→ Explique DEPOIS: Interprete o padrão visual

Tipos recomendados:
- Distribuição → Histograma
- Correlação → Heatmap
- Comparação → Boxplot
- Temporal → Line plot
- Outliers → Scatter + Boxplot

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📄 RELATÓRIO FINAL (Apenas se solicitado explicitamente)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Quando usuário pedir "RELATÓRIO FINAL COMPLETO":

[START_REPORT]

# RELATÓRIO DE ANÁLISE EXPLORATÓRIA DE DADOS

## 1. RESUMO EXECUTIVO
- Síntese em 3-5 linhas
- Principais achados numerados

## 2. CARACTERIZAÇÃO DO DATASET
- Dimensões e tipos de dados
- Qualidade (nulos, duplicatas, inconsistências)
- Estatísticas-chave por coluna

## 3. DESCOBERTAS PRINCIPAIS
### 3.1 Padrões Identificados
- Liste padrões com evidências numéricas

### 3.2 Correlações e Relações
- Correlações fortes com interpretação
- Dependências entre variáveis

### 3.3 Anomalias e Outliers
- Quantidade, localização, possíveis causas
- Impacto nos resultados

## 4. INSIGHTS DE NEGÓCIO
- Tradução de cada descoberta técnica em valor de negócio
- Oportunidades identificadas
- Riscos detectados

## 5. RECOMENDAÇÕES
- Ações prioritárias baseadas em dados
- Próximos passos para investigação
- Melhorias sugeridas

## 6. CONCLUSÃO
- Síntese final objetiva

[END_REPORT]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚀 COMECE AGORA:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Você já analisou o dataset. Aguarde perguntas do usuário.
Responda de forma TÉCNICA, OBJETIVA e com INSIGHTS DE NEGÓCIO.
"""
        )
        
        try:
            st.success(f"[OK] Agente inicializado com {AGENT_CONFIG['model']}")
        except:
            pass

# --- Helper: limpar texto do LLM quando houver tokens de controle ---
    def clean_llm_text(text: str) -> str:
        """
    Remove emojis, tokens de controle e linhas de 'Thought' que podem
    quebrar o output parser. Retorna texto limpo.
    """
        if text is None:
         return ""
    # Remover emoji ⚠️ e qualquer ocorrência similar
        text = re.sub(r"⚠️.*\n?", "", text)
    # Remover o token [LIMIT_REACHED] e a linha inteira
        text = re.sub(r"\[LIMIT_REACHED\].*\n?", "", text, flags=re.IGNORECASE)
    # Remover linhas que começam com "Thought:" (se estiverem presentes)
        text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("Thought:"))
        return text.strip()

    def _prepare_dataset_info(self) -> str:
        """Prepara informações compactas do dataset (string), semelhante ao original."""
        numeric_cols = self.df.select_dtypes(include="number").columns.tolist()
        categorical_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()

        info = (
            f"Dataset carregado:\n"
            f"- Total de linhas: {len(self.df):,}\n"
            f"- Total de colunas: {len(self.df.columns)}\n"
            f"- Colunas numéricas ({len(numeric_cols)}): {', '.join(numeric_cols) if numeric_cols else 'Nenhuma'}\n"
            f"- Colunas categóricas ({len(categorical_cols)}): {', '.join(categorical_cols) if categorical_cols else 'Nenhuma'}\n"
            f"- Memória: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n"
            f"- Valores nulos (total): {int(self.df.isnull().sum().sum())}\n"
            f"- Duplicatas (linhas): {int(self.df.duplicated().sum())}\n\n"
            f"Primeiras linhas:\n{self.df.head(3).to_string()}\n"
        )
        return info

    def ask(self, query: str) -> str:
        """
        Processa pergunta com contexto completo:
        - cria enhanced_query incluindo histórico sumário
        - executa agent.run(...) e trata OutputParserException
        - salva histórico e persiste na memória
        """
        try:
            context = self._get_analysis_summary()

            enhanced_query = f"""
HISTÓRICO:
{context}

PERGUNTA:
{query}

Responda de forma completa usando o contexto quando relevante.
"""
            # Preferimos .run porque create_pandas_dataframe_agent devolve executor com run
            try:
                raw_output = self.agent.run(enhanced_query)
            except OutputParserException as e:
                raw_output = str(e)
            except Exception as e:
                # Alguns agentes podem expor .invoke(...) — fallback
                try:
                    resp = self.agent.invoke({"input": enhanced_query})
                    raw_output = resp.get("output", str(resp))
                except Exception as e2:
                    raise e2

            # Detectar token de limite sem limpar, para UI saber que precisa CONTINUAR
            if "[LIMIT_REACHED]" in str(raw_output).upper() or "⚠️" in str(raw_output):
                result = str(raw_output)
            else:
                result = clean_llm_text(str(raw_output))

            # Salva no histórico (curto)
            self.analysis_history.append({
                "query": query,
                "response": result,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            # Salva na memória (tratamento compatível)
            try:
                if hasattr(self.memory, "save_context"):
                    self.memory.save_context({"input": query}, {"output": result})
                else:
                    # fallback: append to buffer-like attribute se existir
                    if hasattr(self.memory, "buffer"):
                        self.memory.buffer.append({"input": query, "output": result})
            except Exception as e:
                logger.warning("Falha ao salvar na memória: %s", e)

            return result

        except Exception as e:
            error_msg = f"[ERRO] {str(e)}"
            logger.exception("Erro em EDAAgent.ask: %s", e)
            return error_msg

    def _get_analysis_summary(self) -> str:
        """Resumo das últimas 5 análises (texto curto)."""
        if not self.analysis_history:
            return "Nenhuma analise anterior."
        summary = []
        for i, item in enumerate(self.analysis_history[-5:], 1):
            snippet = item["response"][:200].replace("\n", " ")
            summary.append(f"{i}. {item['query']}: {snippet}...")
        return "\n".join(summary)

    def get_conclusions(self) -> str:
        """
        Gera conclusões consolidadas baseadas no histórico.
        Chama o LLM diretamente para gerar o relatório final.
        """
        if not self.analysis_history:
            return "Nenhuma analise realizada. Faça perguntas primeiro!"

        # Constrói o prompt com o histórico completo (cuidado com token length)
        full_summary = f"DATASET:\n{self.dataset_info}\n\nHISTÓRICO COMPLETO ({len(self.analysis_history)} análises):\n"
        for i, item in enumerate(self.analysis_history, 1):
            full_summary += f"\n--- ANALISE {i} ({item['timestamp']}) ---\nPergunta: {item['query']}\nResposta: {item['response']}\n"

        conclusion_prompt = f"""
{full_summary}

Gere relatorio COMPLETO e DETALHADO com:

1. RESUMO EXECUTIVO
2. DESCOBERTAS (padrões, tendências, anomalias)
3. CORRELAÇÕES RELEVANTES
4. QUALIDADE DOS DADOS (nulos, outliers)
5. INSIGHTS E RECOMENDAÇÕES
6. CONCLUSÃO FINAL

Seja ESPECÍFICO com NÚMEROS das análises.
"""
>>>>>>> d4ead596466e6316ed201c1b2862c7a17a1c2125
