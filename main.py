from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

def gerar_roteiro_viagem(numero_dias: int, numero_criancas: int, atividade: str) -> str:
    """
    Gera um roteiro de viagem usando LLM local no LM Studio.
    
    Args:
        numero_dias: Quantidade de dias para o roteiro
        numero_criancas: Quantidade de crianças na família
        atividade: Tipo de atividade que a família gosta
    
    Returns:
        Roteiro de viagem gerado pela LLM
    """
    modelo_prompt = PromptTemplate(
        input_variables=["numero_dias", "numero_criancas", "atividade"],
        template="""
            Crie um roteiro de viagem para um período de {numero_dias} dias, 
            para uma família com {numero_criancas} crianças, que gosta de {atividade}
        """
    )

    prompt = modelo_prompt.format(
        numero_dias=numero_dias,
        numero_criancas=numero_criancas,
        atividade=atividade
    )

    print("Prompt gerado:", prompt)
    
    modelo = ChatOpenAI(
        model="local-model",
        temperature=0.7,
        api_key="lm-studio",
        base_url="http://127.0.0.1:1234/v1"
    )
    
    resposta = modelo.invoke(prompt)
    return resposta.content


if __name__ == "__main__":
    # Parâmetros do roteiro
    numero_dias = 7
    numero_criancas = 2
    atividade = "praias e parques temáticos"
    
    # Gera o roteiro
    roteiro = gerar_roteiro_viagem(numero_dias, numero_criancas, atividade)
    print(roteiro)