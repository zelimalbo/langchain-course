from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


def main():
    print("Hello from langchain-course!")
    information = """
    The BMW 3 Series (E46) is the fourth generation of the BMW 3 Series range of compact executive cars manufactured by German automaker BMW. Produced from 1997 to 2006, it was the successor to the E36 3 Series, which ceased production in 2000. It was introduced in November 1997, and available in sedan, coupé, convertible, station wagon and hatchback body styles. The latter has been marketed as the 3 Series Compact.

    The M3 performance model was introduced in June 2000 with a 2-door coupé body style, followed by the convertible counterpart in April 2001. The M3 is powered by the BMW S54 straight-six engine with either a 6-speed manual or a 6-speed SMG-II automated manual transmission.[15]

    The E46 line-up was phased out starting from late 2004, following the introduction of the E90 3 Series sedans. However, the E46 coupé and convertible body styles remained in production until August 2006.[16]

    """

    summary_template = """
        given the information {information} about a car please tell me:
        1. A short summary
        2. The year it was produced
        3. The engine type
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOllama(temperature=0, model="llama3.2")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(response.content)


if __name__ == "__main__":
    main()