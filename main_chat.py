import os
# Importa load_dotenv para carregar variáveis de ambiente de um arquivo .env
from dotenv import load_dotenv
# Importa ChatGoogleGenerativeAI do langchain_google_genai para interagir com o modelo de linguagem do Google
from langchain_google_genai import ChatGoogleGenerativeAI
# Importa ChatPromptTemplate do langchain_core.prompts para criar prompts de chat
from langchain_core.prompts import ChatPromptTemplate
# Importa StrOutputParser do langchain_core.output_parsers para processar saídas do modelo
from langchain_core.output_parsers import StrOutputParser
# Importa InMemoryChatMessageHistory do langchain_core.memory para armazenar e recuperar conversas
from langchain_core.chat_history import InMemoryChatMessageHistory
# Importa RunnableWithMessageHistory do langchain_core.runnables.history para criar runnables com histórico de mensagens
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    api_key=api_key,
)

prompt_sugestao = ChatPromptTemplate.from_messages(
    [
        ("system", "Você é um assistente de viagem útil especializado em destinos europeus. Apresente-se como Sr. Passeio"),
        ("placeholder", "{historico}"),
        ("human", "{query}"),
    ]
)

cadeia = prompt_sugestao | modelo | StrOutputParser()

memoria = {}
sessao = "meu_id_de_sessao"

def historico_conversa(sessao_id: str) -> str:
    if sessao_id not in memoria:
        memoria[sessao_id] = InMemoryChatMessageHistory()
    return memoria[sessao_id]

perguntas = [
    'Quero visitar um lugar na Europa, famoso por suas catedrais góticas. Qual cidade você recomendaria?',
    'Qual a melhor época do ano para ir?'
]

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_conversa,
    input_messages_key="query",
    history_messages_key="historico",
)

for pergunta in perguntas:
    resposta = cadeia_com_memoria.invoke(
        {
            "query": pergunta,
        },
        config={"session_id": sessao}
    )
    print(f"Pergunta: {pergunta}\nResposta: {resposta}\n")