import os

from dotenv import load_dotenv

from langchain import PromptTemplate
from langchain.agents import ConversationalChatAgent, load_tools, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage

import streamlit as st
from streamlit_chat import message

#
# Load environment variables
#

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


system_message = SystemMessage(
    content="""You are chatting with an AI. The AI is very smart and can talk about anything."""
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}


# llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

llm = ChatOpenAI(
    model="gpt-4",
)

tools = load_tools(["ddg-search"])

if "memory" not in st.session_state:
    memory = ConversationSummaryBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        llm=llm,
        max_token_limit=1000,
    )
    st.session_state["memory"] = memory

if "past_user" not in st.session_state:
    st.session_state["past_user"] = []
if "past_ai" not in st.session_state:
    st.session_state["past_ai"] = []


def main():
    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=llm,
        tools=tools,
    )

    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=st.session_state["memory"],
    )

    st.set_page_config(page_title="ai2ai", page_icon="🤖", layout="wide")

    st.title("🤖💬🗨️🤖")

    if prompt := st.chat_input():
        with st.spinner("Thinking..."):
            st_callback = StreamlitCallbackHandler(st.container())
            response = agent_chain.run(input=prompt, callbacks=[st_callback])

        st.session_state["past_user"].append(prompt)
        st.session_state["past_ai"].append(response)

        for i in range(len(st.session_state["past_user"])):
            message(
                st.session_state["past_user"][i],
                key=str(i),
                is_user=True,
                avatar_style="lorelei-neutral",
                seed="Harley",
            )

            message(
                st.session_state["past_ai"][i],
                key=str(i) + "_ai",
                is_user=False,
            )


if __name__ == "__main__":
    main()
