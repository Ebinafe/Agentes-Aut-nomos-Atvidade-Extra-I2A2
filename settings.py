# CÓDIGO CORRIGIDO PARA src/config/settings.py

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv 


# Encontra a raiz do projeto
DOTENV_PATH = find_dotenv()

# Carrega variáveis de ambiente
load_dotenv(dotenv_path=DOTENV_PATH, override=True)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Obtém a chave e remove espaços
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if OPENROUTER_API_KEY:
    OPENROUTER_API_KEY = OPENROUTER_API_KEY.strip()

# Validação (código omitido, mas você deve mantê-lo)

# Lista de modelos em ordem de preferência (fallback automático)
# AGORA APENAS COM O GPT-4o MINI
MODEL_PRIORITY = [
    {
        "name": "GPT-4o Mini",
        "model": "openai/gpt-4o-mini",
        "cost": "$0.15/1M entrada",
        "context": "128k tokens",
        "temperature": 0.1,
        "max_tokens": 8000,
    },
]

# Configuração do modelo padrão (primeiro e único da lista)
AGENT_CONFIG = MODEL_PRIORITY[0]

# URL base do OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Configurações de processamento (Mantenha as suas)
MAX_ROWS_DISPLAY = 1000
CHUNK_SIZE = 50000

# Configurações de retry (Ajustadas para 1, pois só há 1 modelo)
MAX_RETRIES = len(MODEL_PRIORITY)  # Tenta apenas o modelo único
RETRY_DELAY = 2  # segundos entre tentativas