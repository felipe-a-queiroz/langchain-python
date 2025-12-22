# Importanto ChatGoogleGenerativeAI da biblioteca langchain_google_genai para interagir com o modelo de linguagem do Google
from langchain_google_genai import ChatGoogleGenerativeAI
# Importanto ChatPromptTemplate da biblioteca langchain_core.prompts para criar prompts de chat
from langchain_core.prompts import ChatPromptTemplate
# Importanto StrOutputParser da biblioteca langchain_core.output_parsers para processar saídas do modelo
from langchain_core.output_parsers import StrOutputParser
# Importando load_dotenv para carregar variáveis de ambiente de um arquivo .env
from dotenv import load_dotenv
# Importando os para acessar variáveis de ambiente
import os

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

# Configurando o modelo de linguagem do Google com parâmetros específicos
modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    api_key=api_key,
)

# Definindo o prompt para o consultor de viagens
prompt_consultor = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um consultor de viagens"),
        ("human", "{query}"),
    ],
)

# Combinando o prompt com o modelo e o parser de saída
assistente = prompt_consultor | modelo | StrOutputParser()

print(assistente.invoke({"query": "Quais são os melhores destinos para visitar na Europa?"}))
