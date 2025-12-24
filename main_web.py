import os
from dotenv import load_dotenv
from langchain_core.globals import set_debug
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
set_debug(True)

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

url = "https://forbes.com.br/forbes-tech/2025/08/o-que-explica-o-fracasso-do-chatgpt-5-e-como-a-openai-vai-reagir/"
loader = WebBaseLoader(url)
documento = loader.load()

chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
).split_documents(documento)

dados_recuperados = FAISS.from_documents(
    chunks, embeddings
).as_retriever(search_kwargs={"k":2})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Responda usando exclusivamente o conteúdo fornecido"),
    ("human", "{query}\n\nContexto: \n{contexto}\n\nResposta:")
])

cadeia = prompt | modelo | StrOutputParser()

def responder_pergunta(pergunta: str) -> str:
    print(f"Pergunta: {pergunta}")
    documentos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join([doc.page_content for doc in documentos])
    resposta = cadeia.invoke({
        "query": pergunta,
        "contexto": contexto
    })
    return resposta

query = "Qual o motivo apresentado para o fracasso do GPT-5?"
resposta = responder_pergunta(query)
print(resposta)