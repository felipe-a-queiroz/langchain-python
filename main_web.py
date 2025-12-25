import os
from dotenv import load_dotenv
from langchain_core.globals import set_debug
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
set_debug(False)

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    api_key=api_key,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=api_key,
)

url = "https://www.centrodombosco.org/artigos/dom-athanasius-schneider-contesta-nota-doutrinal-do-dicasterio-e-reafirma-corredentora"
loader = WebBaseLoader(url)
documento = loader.load()

chunks_por_caracter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
).split_documents(documento)

print(f"Número de chunks por caracter: {len(chunks_por_caracter)}")
for i, chunk in enumerate(chunks_por_caracter):
    print(f"--- Chunk por caracter {i+1} ---")
    print(chunk.page_content)
    print("\n")

chunks_por_token = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=300,
    chunk_overlap=0
).split_documents(documento)

print(f"Número de chunks por token: {len(chunks_por_token)}")
for i, chunk in enumerate(chunks_por_token):
    print(f"--- Chunk por token {i+1} ---")
    print(chunk.page_content)
    print("\n")

dados_recuperados_por_caracteres = FAISS.from_documents(
    chunks_por_token, embeddings
).as_retriever(search_kwargs={"k":2})

dados_recuperados_por_tokens = FAISS.from_documents(
    chunks_por_token, embeddings
).as_retriever(search_kwargs={"k":2})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Responda usando exclusivamente o conteúdo fornecido"),
    ("human", "{query}\n\nContexto: \n{contexto}\n\nResposta:")
])

cadeia = prompt | modelo | StrOutputParser()

def responder_pergunta_por_caracteres(pergunta: str) -> None:
    documentos = dados_recuperados_por_caracteres.invoke(pergunta)
    contexto = "\n\n".join([doc.page_content for doc in documentos])
    resposta = cadeia.invoke({
        "query": pergunta,
        "contexto": contexto
    })
    print(f"--- Resposta por caracteres: {resposta}")
    print("\n")

def responder_pergunta_por_tokens(pergunta: str) -> None:
    documentos = dados_recuperados_por_tokens.invoke(pergunta)
    contexto = "\n\n".join([doc.page_content for doc in documentos])
    resposta = cadeia.invoke({
        "query": pergunta,
        "contexto": contexto
    })
    print(f"--- Resposta por tokens: {resposta}")
    print("\n")

perguntas = [
    "Qual a opinião dos primeiros padres da igreja sobre o papel da Virgem Maria na redenção?",
    "O que significa o título 'Corredentora' atribuído à Virgem Maria?",
]
for i, query in enumerate(perguntas):
    print(f"\n\nPergunta {i+1}: {query}")
    responder_pergunta_por_caracteres(query)
    responder_pergunta_por_tokens(query)