from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Definindo o prompt para sugerir uma cidade com base no interesse do usuário
prompt_cidade = PromptTemplate(
    template="""
        Sugira uma cidade dado o meu interesse em {interesse}.
    """,
    input_variables=["interesse"]

)

# Definindo o interesse do usuário
interesse = "tecnologia e inovação"

# Configurando o modelo de linguagem local
modelo = ChatOpenAI(
    model="local-model",
    temperature=0.7,
    api_key="lm-studio",
    base_url="http://127.0.0.1:1234/v1"
)

# Criando a cadeia de processamento com LCEL
cadeia = prompt_cidade | modelo | StrOutputParser()

# Invocando a cadeia com o interesse do usuário
resposta = cadeia.invoke(
    {
        "interesse": interesse
    }
)

# Exibindo a resposta do modelo
print("Resposta do modelo:", resposta)