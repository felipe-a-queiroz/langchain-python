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
# Importando Literal e TypeDict para definir tipos literais em modelos de dados
from typing import Literal, TypedDict


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

# Definindo o prompt para o consultor de praia
prompt_consultor_praia = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sra. Praia. Você é um consultor de viagens especialista em destinos de praia"),
        ("human", "{query}"),
    ],
)

# Definindo o prompt para o consultor de montanha
prompt_consultor_montanha = ChatPromptTemplate.from_messages(
    [
        ("system", "Apresente-se como Sr. Montanha. Você é um consultor de viagens especialista em destinos de montanha"),
        ("human", "{query}"),
    ],
)

cadeia_praia = prompt_consultor_praia | modelo | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | modelo | StrOutputParser()


class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

prompt_roteador = ChatPromptTemplate.from_messages(
    [
        ("system", "Responda apenas com 'praia' ou 'montanha'"),
        ("human", "{query}"),
    ],
)

# Definindo o roteador que decide qual consultor usar com base na consulta do usuário
roteador = prompt_roteador | modelo.with_structured_output(Rota)

# Função para processar a consulta do usuário e direcioná-la ao consultor apropriado, sem usar o LangGraph
def response(query: str) -> str:
    rota = roteador.invoke({"query": query})["destino"]
    if rota == "praia":
        return cadeia_praia.invoke({"query": query})
    else:
        return cadeia_montanha.invoke({"query": query})
    
print(response("Quero surfar em um lugar de águas cristalinas."))