import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    api_key=api_key,
)

perguntas = [
    'Quero visitar um lugar na Europa, famoso por suas catedrais góticas. Qual cidade você recomendaria?',
    'Qual a melhor época do ano para ir?'
]

for pergunta in perguntas:
    resposta = modelo.invoke(pergunta)
    print(f"Pergunta: {pergunta}\nResposta: {resposta.content}\n")