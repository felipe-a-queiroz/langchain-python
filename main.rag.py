from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.5,
    api_key=api_key,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=api_key,
)

# Pipeline de ingestão de documentos
## Extração do documento
documento = TextLoader(
    "documentos/GTB_gold_Nov23.txt",
    encoding="utf-8",
).load()

## Divisão do documento em pedaços menores
chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
).split_documents(documento)

## Faz o embedding e carrega os pedaços no FAISS
dados_recuperados = FAISS.from_documents(
    chunks, embeddings
).as_retriever(search_kwargs={"k":2})

prompt_consulta_seguro = ChatPromptTemplate.from_messages([
    ("system", "Responda usando exclusivamente o conteúdo fornecido"),
    ("human", "{query}\n\nContexto: \n{contexto}\n\nResposta:")
])

cadeia = prompt_consulta_seguro | modelo | StrOutputParser()

def responder_pergunta(pergunta: str) -> str:
    documents = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join([doc.page_content for doc in documents])
    resposta = cadeia.invoke({
        "query": pergunta,
        "contexto": contexto
    })
    return resposta

print(responder_pergunta("Como devo proceder caso tenha um item roubado?"))