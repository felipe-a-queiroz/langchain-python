# Importa ChatGoogleGenerativeAI do langchain_google_genai para interagir com o modelo de linguagem do Google
from langchain_google_genai import ChatGoogleGenerativeAI
# Importa PromptTemplate do langchain_core para criar templates de prompts
from langchain_core.prompts import PromptTemplate
# Importa JsonOutputParser e StrOutputParser do langchain_core para processar saídas do modelo
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
# Importa Field do Pydantic para definir campos em modelos de dados
# Importa BaseModel do Pydantic para criar modelos de dados estruturados
from pydantic import Field, BaseModel
# Importa set_debug do langchain_core.globals para configurar o modo de depuração
from langchain_core.globals import set_debug
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

# Ativando o modo de depuração para obter mais informações durante a execução
set_debug(True)

# Definindo o modelo de dados para a cidade sugerida
class Destino(BaseModel):
    cidade: str = Field(description="Nome da cidade sugerida")
    motivo: str = Field(description="Motivo pelo qual a cidade foi sugerida")

# Definindo o modelo de dados para a lista de restaurantes
class Restaurantes(BaseModel):
    cidade: str = Field(description="A cidade recomendada para visitar")
    restaurantes: str = Field(description="Lista de restaurantes recomendados na cidade")

# Definindo o parser de saída para converter a resposta do modelo em um objeto Pydantic
parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurantes = JsonOutputParser(pydantic_object=Restaurantes)

# Definindo o prompt para sugerir uma cidade com base no interesse do usuário
prompt_cidade = PromptTemplate(
    template="""
        Sugira um nome de cidade real dado o meu interesse em {interesse}. 
        {formato_de_saida}
    """,
    input_variables=["interesse"],
    output_parser=parseador_destino,
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions()},
)

prompt_restaurantes = PromptTemplate(
    template="""
        Sugira uma lista de restaurantes em {cidade}.
        {formato_de_saida}
    """,
    input_variables=["cidade"],
    output_parser=parseador_restaurantes,
    partial_variables={"formato_de_saida": parseador_restaurantes.get_format_instructions()},
)

prompt_cultural = PromptTemplate(
    template="""
        Sugira atividades e locais culturais para visitar em {cidade}.
    """,
    input_variables=["cidade"],
)

# Configurando o modelo de linguagem local
modelo = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7,
    api_key=api_key,
)


# Criando a cadeia de processamento com LCEL
cadeia_1 = prompt_cidade | modelo | parseador_destino
cadeia_2 = prompt_restaurantes | modelo | parseador_restaurantes
cadeia_3 = prompt_cultural | modelo | StrOutputParser()

cadeia = (cadeia_1 | cadeia_2 | cadeia_3)

# Invocando a cadeia com o interesse do usuário
resposta = cadeia.invoke(
    {
        "interesse": "catolicismo e arquitetura gótica",
    }
)

# Exibindo a resposta do modelo
print("Resposta do modelo:", resposta)