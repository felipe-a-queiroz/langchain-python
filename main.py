# Importa ChatOpenAI do langchain_openai para interagir com modelos de linguagem
from langchain_openai import ChatOpenAI
# Importa PromptTemplate do langchain_core para criar templates de prompts
from langchain_core.prompts import PromptTemplate
# Importa JsonOutputParser e StrOutputParser do langchain_core para processar saídas do modelo
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
# Importa Field do Pydantic para definir campos em modelos de dados
# Importa BaseModel do Pydantic para criar modelos de dados estruturados
from pydantic import Field, BaseModel
# Importa set_debug do langchain_core.globals para configurar o modo de depuração
from langchain_core.globals import set_debug

# Ativando o modo de depuração para obter mais informações durante a execução
set_debug(True)

# Definindo o modelo de dados para a resposta esperada
class Destino(BaseModel):
    cidade: str = Field(description="Nome da cidade sugerida")
    motivo: str = Field(description="Motivo pelo qual a cidade foi sugerida")

# Definindo o parser de saída para converter a resposta do modelo em um objeto Pydantic
parseador = JsonOutputParser(pydantic_object=Destino)

# Definindo o prompt para sugerir uma cidade com base no interesse do usuário
prompt_cidade = PromptTemplate(
    template="""
        Sugira uma cidade dado o meu interesse em {interesse}.
        {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador.get_format_instructions()},
)

# Definindo o interesse do usuário
interesse = "catolicismo e arquitetura gótica europeia"

# Configurando o modelo de linguagem local
modelo = ChatOpenAI(
    model="local-model",
    temperature=0.7,
    api_key="lm-studio",
    base_url="http://127.0.0.1:1234/v1"
)


# Criando a cadeia de processamento com LCEL
cadeia = prompt_cidade | modelo | parseador

# Invocando a cadeia com o interesse do usuário
resposta = cadeia.invoke(
    {
        "interesse": interesse
    }
)

# Exibindo a resposta do modelo
print("Resposta do modelo:", resposta)