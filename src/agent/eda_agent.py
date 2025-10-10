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
        self.agent = create_pandas_dataframe_agent (
            llm=self.llm,
            df=self.df,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            allow_dangerous_code=False,
            max_iterations=80,
            handle_parsing_errors=True,
            prefix=f"""Você é um Agente Especialista em EDA (Exploratory Data Analysis) focado em **dados financeiros**.
            Seu objetivo: **Responder diretamente** ao usuário com resultados numéricos, tabelas e/ou gráficos, sem mostrar código Python, células, ou instruções de execução.            

        INFORMACOES DO DATASET:
        {self.dataset_info}

        1) SAÍDA: NUNCA apresente código. APRESENTE RESULTADOS:
            - Texto objetivo (máx 300 palavras por resposta curta).
            - Quando apropriado, inclua uma tabela em formato JSON (veja esquema abaixo).
            - Quando solicitado, gere gráficos — mas **no texto de saída** apenas inclua: (a) descrição do gráfico, (b) um bloco JSON com os dados agregados prontos para plot, e (c) interpretação do gráfico. NÃO inclua código de plotagem.

        2) FORMATO PADRÃO (prioritário):
            - Para respostas diretas: texto conciso com **valores exatos** (até 2 casas decimais).
            - Para tabelas/visualizações: forneça um objeto JSON com o formato:
        {
       "type": "table" | "plot" | "text" | "report",
       "title": "Título curto",
       "summary": "Uma linha resumo",
       "data":  [ {'x_coord': 10, 'y_coord': 50}, {...} ]
        }
            - Sempre entregue também o "impacto de negócio" (1-2 frases).

        3) MÉTODOS E TRANSPARÊNCIA:
            - Resuma a metodologia em UMA LINHA (ex: "Outliers detectados por IQR (Q1-1.5×IQR, Q3+1.5×IQR)").
            - Cite contagens, médias, percentuais e coeficientes de correlação quando relevantes.

        4) EXEMPLOS DE RESPOSTA (prioridade, não opcional):
            - Pergunta: "Existem outliers em 'amount'?"
            Resposta (texto): "Método: IQR. Resultado: 1.203 outliers (0.85%). Colunas: amount. Impacto: transações > 5.000 representam 0.4% do volume; revisar regras anti-fraude."
            Resposta (JSON): {"type":"table","title":"Outliers_amount","summary":"Outliers por faixa","data":[{"range":">5000","count":324},{"range":"1000-5000","count":879}]}

        5) GRÁFICOS: quando pedido "Mostrar gráfico X":
            - Responda: "Gerando [tipo de gráfico] para [objetivo]" + forneça **dados agregados** em `data` no esquema JSON. Inclua interpretação (2-3 frases).
            - Ex.: {"type":"plot","title":"Volume por mês","data":[{"month":"2024-01","volume":12345},...]}

        6) RELATÓRIO FINAL:
            - Se o usuário pedir "RELATÓRIO FINAL COMPLETO", use o template [START_REPORT] já definido no sistema. Saída final deve ser do tipo `"report"` no formato JSON e também em texto legível.

        7) RUÍDO & CONTROLE:
            - Nunca use frases vagas ("alguns", "vários", "pode ser"). Dê números.
            - Se você não conseguiu calcular por limite de tokens ou dados ausentes, responda objetivamente: "Não foi possível calcular X porque [motivo]." e proponha o passo mínimo a executar.

        8) COMPACTAÇÃO DE TOKENS:
            - Prefira um JSON conciso com campos essenciais quando for possível.
            - Use 2 casas decimais para números agregados; conte exatos para contagens.

        FIM..
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