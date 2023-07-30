import os
from typing import Literal

from dotenv import load_dotenv

from langchain.agents import ConversationalChatAgent, load_tools, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import OutputParserException

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
st.set_page_config(page_title="Red v Blue", page_icon="🤖", layout="wide")

col1, col2 = st.columns([0.2, 0.8])
with col1:
    st.markdown(
        '<img src="./app/static/red-v-blue.png" height="100" style="">',
        unsafe_allow_html=True,
    )
with col2:
    st.write("### :red[Red] v :blue[Blue]")
    st.write("Human-mediated AI agent-to-agent chat")

st.divider()
# st.write(
#     "[![Star](https://img.shields.io/github/stars/witt3rd/gai-ai2ai.svg?logo=github&style=social)](https://gitHub.com/witt3rd/gai-ai2ai)"
#     + "[![Follow](https://img.shields.io/twitter/follow/dt_public?style=social)](https://www.twitter.com/dt_public)"
# )

speakers = {"Red": "red_agent", "Blue": "blue_agent"}
if "first_speaker" not in st.session_state:
    st.session_state["first_speaker"] = "Red"

with st.expander("Dialog Configuration", expanded=True):
    with st.form("agent_config"):
        col1, col2 = st.columns(2)

        with col1:
            red_directive = st.text_area(":red[🤖 Red Directive]")

        with col2:
            blue_directive = st.text_area(":blue[🤖 Blue Directive]")

        speaker = st.selectbox("First speaker", list(speakers.keys()), index=0)

        update = st.form_submit_button("Update", use_container_width=True)
        if update:
            # Validate
            if not red_directive:
                st.error("Red directive is required")
                st.stop()
            if not blue_directive:
                st.error("Blue directive is required")
                st.stop()

            # Update the state
            has_changed = False
            if (
                "red_directive" in st.session_state
                and st.session_state["red_directive"] != red_directive
            ) or "red_directive" not in st.session_state:
                has_changed = True
                st.session_state["red_directive"] = red_directive

            if (
                "blue_directive" in st.session_state
                and st.session_state["blue_directive"] != blue_directive
            ) or "blue_directive" not in st.session_state:
                has_changed = True
                st.session_state["blue_directive"] = blue_directive

            if (
                "first_speaker" in st.session_state
                and st.session_state["first_speaker"] != speaker
            ) or "first_speaker" not in st.session_state:
                has_changed = True
                st.session_state["first_speaker"] = speaker

            # if anything has changed, reset the session state
            if has_changed:
                if "red_agent" in st.session_state:
                    del st.session_state["red_agent"]
                if "blue_agent" in st.session_state:
                    del st.session_state["blue_agent"]
                if "chat_history" in st.session_state:
                    del st.session_state["chat_history"]
                if "memory" in st.session_state:
                    del st.session_state["memory"]
                if "response" in st.session_state:
                    del st.session_state["response"]
                st.session_state["speaker"] = st.session_state["first_speaker"]
                st.success("Updated")
            else:
                st.info("No changes")

if not "red_directive" in st.session_state or not "blue_directive" in st.session_state:
    st.stop()

#
# Session state
#

if "llm" not in st.session_state:
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.0,
    )
    st.session_state["llm"] = llm

if "tools" not in st.session_state:
    tools = load_tools(["ddg-search"])
    st.session_state["tools"] = tools

if "memory" not in st.session_state:
    memory = ConversationSummaryBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        llm=st.session_state["llm"],
        max_token_limit=1000,
    )
    st.session_state["memory"] = memory


def create_agent(name: str):
    directive = st.session_state[f"{name}_directive"]
    agent = ConversationalChatAgent.from_llm_and_tools(
        llm=st.session_state["llm"],
        tools=st.session_state["tools"],
        system_message=directive,
    )
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=st.session_state["tools"],
        verbose=True,
        memory=st.session_state["memory"],
    )
    st.session_state[f"{name}_agent"] = agent_chain


if "red_agent" not in st.session_state:
    create_agent("red")

if "blue_agent" not in st.session_state:
    create_agent("blue")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

#
# Chat history
#


def speaker_color(speaker) -> Literal["red", "blue"]:
    color = "red" if speaker == "Red" else "blue"
    return color


print(f'Render: {st.session_state["chat_history"]}')
for i in range(len(st.session_state["chat_history"])):
    turn = st.session_state["chat_history"][i]
    speaker = turn["speaker"]
    color = speaker_color(speaker)
    st.text_area(
        f":{color}[{speaker}]",
        value=turn["content"],
        key=f"chat_history_{i}",
    )


with st.form("chat_input"):
    speaker = st.session_state["speaker"]
    responder = "Red" if speaker == "Blue" else "Blue"
    print(f"speaker: {speaker}, responder: {responder}")
    responder_agent = st.session_state[speakers[responder]]

    color = speaker_color(speaker)
    prompt = st.text_area(
        f":{color}[{speaker}]",
        value=st.session_state["response"] if "response" in st.session_state else "",
    )
    submit = st.form_submit_button("Send", use_container_width=True)
    if submit:
        st.session_state["chat_history"].append({"speaker": speaker, "content": prompt})
        print(st.session_state["chat_history"])
        with st.spinner("Thinking..."):
            st_callback = StreamlitCallbackHandler(st.container())

            try:
                response = responder_agent.run(input=prompt, callbacks=[st_callback])
            except OutputParserException as e:
                response = str(e)
                prefix = "Could not parse LLM output: "
                if not response.startswith(prefix):
                    raise e
                response = response.removeprefix(prefix)

            st.session_state["response"] = response
            st.session_state["speaker"] = responder

            st.experimental_rerun()
