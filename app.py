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

#
# Streamlit app
#
st.set_page_config(page_title="ai2ai", page_icon="🤖", layout="wide")

col1, col2 = st.columns([0.2, 0.8])
with col1:
    st.markdown(
        '<img src="./app/static/bot-talk.png" height="100" style="">',
        unsafe_allow_html=True,
    )
with col2:
    st.write("### ai2ai")
    st.write("Human-mediated AI agent-to-agent chat")

st.divider()
# st.write(
#     "[![Star](https://img.shields.io/github/stars/witt3rd/gai-ai2ai.svg?logo=github&style=social)](https://gitHub.com/witt3rd/gai-ai2ai)"
#     + "[![Follow](https://img.shields.io/twitter/follow/dt_public?style=social)](https://www.twitter.com/dt_public)"
# )

with st.expander("Agent Configuration", expanded=True):
    with st.form("agent_config"):
        col1, col2 = st.columns(2)

        with col1:
            agent_1_directive = st.text_area(
                "Agent 1 Directive",
            )

        with col2:
            agent_2_directive = st.text_area("Agent 2 Directive")

        update = st.form_submit_button("Update", use_container_width=True)
        if update:
            # Validate
            if not agent_1_directive:
                st.error("Agent 1 Directive is required")
                st.stop()
            if not agent_2_directive:
                st.error("Agent 2 Directive is required")
                st.stop()

            # Update the state
            has_changed = False
            if (
                "agent_1_directive" in st.session_state
                and st.session_state["agent_1_directive"] != agent_1_directive
            ) or "agent_1_directive" not in st.session_state:
                has_changed = True
                st.session_state["agent_1_directive"] = agent_1_directive
            if (
                "agent_2_directive" in st.session_state
                and st.session_state["agent_2_directive"] != agent_2_directive
            ) or "agent_2_directive" not in st.session_state:
                has_changed = True
                st.session_state["agent_2_directive"] = agent_2_directive

            if has_changed:
                del st.session_state["agent_1"]
                del st.session_state["agent_2"]
                del st.session_state["chat_history"]
                st.success("Updated")
            else:
                st.info("No changes")

if (
    not "agent_1_directive" in st.session_state
    or not "agent_2_directive" in st.session_state
):
    st.stop()

#
# Session state
#

if "llm" not in st.session_state:
    llm = ChatOpenAI(
        model="gpt-4",
    )
    st.session_state["llm"] = llm

if "tools" not in st.session_state:
    tools = load_tools(["ddg-search"])
    st.session_state["tools"] = tools

if "memory" not in st.session_state:
    memory = ConversationSummaryBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        llm=llm,
        max_token_limit=1000,
    )
    st.session_state["memory"] = memory


def create_agent(name: str):
    directive = st.session_state[f"{name}_directive"]
    print(f"Directive: {directive}")
    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=st.session_state["llm"],
        tools=st.session_state["tools"],
        system_message=directive,
    )
    print(f"Prompt: {agent.llm_chain.prompt}")
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=st.session_state["tools"],
        verbose=True,
        memory=st.session_state["memory"],
    )
    print(f"Created agent: {name}")
    st.session_state[name] = agent_chain


if "agent_1" not in st.session_state:
    create_agent("agent_1")

if "agent_2" not in st.session_state:
    create_agent("agent_2")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

#
# Render
#
talker = st.selectbox("Agent", ["Agent 1", "Agent 2"])

if prompt := st.chat_input():
    with st.spinner("Thinking..."):
        st_callback = StreamlitCallbackHandler(st.container())

        agent = (
            st.session_state["agent_1"]
            if talker == "Agent 1"
            else st.session_state["agent_2"]
        )
        if not agent:
            st.error("Agent not found")
            st.stop()
        response = agent.run(input=prompt, callbacks=[st_callback])

    turn = {
        "prompt": prompt,
        "response": response,
    }
    st.session_state["chat_history"].append(turn)

for i in range(len(st.session_state["chat_history"])):
    turn = st.session_state["chat_history"][i]
    message(
        turn["prompt"],
        key=str(i),
        is_user=True,
        avatar_style="lorelei-neutral",
        seed="Harley",
    )

    message(
        turn["response"],
        key=str(i) + "_ai",
        is_user=False,
    )
