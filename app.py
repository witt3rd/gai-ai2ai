import os

import chainlit as cl
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper

load_dotenv()

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0, model="gpt-4", streaming=True)

search = DuckDuckGoSearchRun()
wikipedia = WikipediaAPIWrapper()

# Web Search Tool
search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="A useful tool for searching the Internet to find information on world events, issues, etc. Worth using for general topics. Use precise questions.",
)

# Wikipedia Tool
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="A useful tool for encyclopedic articles on specific concepts, people, places, things, events, etc. Use precise questions.",
)

prompt = PromptTemplate(
    template="""Plan: {input}

History: {chat_history}

Let's think about the answer step by step.
If it's information retrieval task, solve it like a professor in a particular field.""",
    input_variables=["input", "chat_history"],
)

plan_prompt = PromptTemplate(
    input_variables=["input", "chat_history"],
    template="""Prepare plan for task execution. (e.g. retrieve current date to find weather forecast)

    Tools to use: wikipedia, web search

    REMEMBER: Keep in mind that you don't have information about current date, temperature, informations after September 2021. Because of that you need to use tools to find them.

    Question: {input}

    History: {chat_history}

    Output look like this:
    '''
        Question: {input}

        Execution plan: [execution_plan]

        Rest of needed information: [rest_of_needed_information]
    '''

    IMPORTANT: if there is no question, or plan is not need (YOU HAVE TO DECIDE!), just populate {input} (pass it as a result). Then output should look like this:
    '''
        input: {input}
    '''
    """,
)

@cl.on_chat_start
def main():

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


    plan_chain = ConversationChain(
        llm=llm,
        memory=memory,
        callbacks=[cl.AsyncLangchainCallbackHandler()],
        input_key="input",
        prompt=plan_prompt,
        output_key="output",
    )

    cl.user_session.set("plan_chain", plan_chain)

    # Initialize Agent
    agent = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=[search_tool, wikipedia_tool],
        llm=llm,
        verbose=True, # verbose option is for printing logs (only for development)
        max_iterations=3,
        prompt=prompt,
        memory=memory,
    )

    cl.user_session.set("agent", agent)

@cl.on_message
async def main(message: str):
    plan_chain = cl.user_session.get("plan_chain")
    agent = cl.user_session.get("agent")

    # Plan execution
    plan_result = await plan_chain.acall(message)
    result = plan_result["output"]

    # Agent execution
    cb = cl.LangchainCallbackHandler()
    answer = await cl.make_async(agent.run)(result, callbacks=[cb])
    await cl.Message(author="Agent", content=answer).send()
