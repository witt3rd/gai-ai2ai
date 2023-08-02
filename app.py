from datetime import datetime
import json
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain.agents import ConversationalChatAgent, load_tools, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler, get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationSummaryBufferMemory
from langchain.memory import ConversationBufferMemory
from langchain.schema import (
    AIMessage,
    BaseMemory,
    HumanMessage,
    OutputParserException,
)

import streamlit as st
from streamlit_chat import message

#
# Custom memory that is shared between two agents
#


class SharedMemory(BaseMemory, BaseModel):
    """Memory class for storing information about entities."""

    chat_history: ChatMessageHistory = ChatMessageHistory()

    # Define key to pass information about entities into prompt.
    memory_key: str = "chat_history"

    def clear(self):
        self.chat_history = ChatMessageHistory()

    @property
    def memory_variables(self) -> list[str]:
        """Define the variables we are providing to the prompt."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, any]) -> dict[str, str]:
        """Load the memory variables, in this case the entity key."""
        # print(f"Loading memory variables: {inputs}")
        return {self.memory_key: self.chat_history.messages}

    def save_context(self, inputs: dict[str, any], outputs: dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        # text = inputs[list(inputs.keys())[0]]
        # print(f"Saving context: {inputs} -> {outputs}")


#
# Load environment variables
#

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = os.getenv("DATA_DIR", "./data")

#
# Scenario library
#

with open("scenario_library.json", "r") as f:
    scenario_library = json.load(f)

#
# Chat sessions
#


def session_create(
    scenario: str,
    red_bot: str,
    red_directive: str,
    blue_bot: str,
    blue_directive: str,
    first_speaker: str,
) -> (str, dict[str, any]):
    session = {
        "scenario": scenario,
        "red_bot": red_bot,
        "red_directive": red_directive,
        "blue_bot": blue_bot,
        "blue_directive": blue_directive,
        "first_speaker": first_speaker,
        "messages": [],
    }
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_file = os.path.join(DATA_DIR, timestamp + ".json")
    session_write(session_file, session)

    return session_file, session


def session_write(
    session_file: str,
    session: dict[str, any],
) -> None:
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    with open(session_file, "w") as f:
        json.dump(session, f, indent=2)


def session_add_message(
    session_file: str,
    session: dict[str, any],
    speaker: str,
    message: str,
) -> None:
    session["messages"].append(
        {
            "speaker": speaker,
            "message": message,
        }
    )
    session_write(session_file, session)


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

speakers = ["Red", "Blue"]


def speaker_color(speaker) -> Literal["red", "blue"]:
    color = "red" if speaker == "Red" else "blue"
    return color


def other_speaker(speaker) -> Literal["Red", "Blue"]:
    other = "Red" if speaker == "Blue" else "Blue"
    return other


# Reset the chat on first speaker change
def reset_dialog() -> None:
    if "speaker" in st.session_state:
        del st.session_state["speaker"]
    if "Red" in st.session_state:
        del st.session_state["Red"]
    if "Blue" in st.session_state:
        del st.session_state["Blue"]
    if "memory" in st.session_state:
        del st.session_state["memory"]
    if "response" in st.session_state:
        del st.session_state["response"]
    if "tokens" in st.session_state:
        del st.session_state["tokens"]
    if "cost" in st.session_state:
        del st.session_state["cost"]


if "current_scenario_key" not in st.session_state:
    st.session_state["current_scenario_key"] = scenario_library["current_scenario_key"]

with st.expander("Directives", expanded=True):
    current_scenario_key = st.session_state["current_scenario_key"]
    current_scenario = scenario_library["scenarios"][current_scenario_key]
    first_speaker = current_scenario["first_speaker"]
    scenarios = list(scenario_library["scenarios"])
    bots = list(scenario_library["bots"])

    st.selectbox(
        "Scenario",
        scenarios,
        key="current_scenario_key",
        index=scenarios.index(current_scenario_key),
        on_change=reset_dialog,
    )

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.selectbox(
                ":red[🤖 Red Bot]",
                bots,
                key="red_bot",
                index=bots.index(current_scenario["Red"]),
                on_change=reset_dialog,
            )
            st.text_area(
                ":red[🤖 Red Directive]",
                value=scenario_library["bots"][st.session_state["red_bot"]],
                key="red_directive",
                on_change=reset_dialog,
            )

        with col2:
            st.selectbox(
                ":blue[🤖 Blue Bot]",
                bots,
                key="blue_bot",
                index=bots.index(current_scenario["Blue"]),
                on_change=reset_dialog,
            )
            st.text_area(
                ":blue[🤖 Blue Directive]",
                value=scenario_library["bots"][st.session_state["blue_bot"]],
                key="blue_directive",
                on_change=reset_dialog,
            )

        st.selectbox(
            "First speaker",
            speakers,
            index=speakers.index(first_speaker),
            key="first_speaker",
            on_change=reset_dialog,
        )

        # update = st.form_submit_button("Update", use_container_width=True)
        # if update:
        #     # Validate
        #     if not st.session_state["red_directive"]:
        #         st.error("Red directive is required")
        #         st.stop()
        #     if not st.session_state["blue_directive"]:
        #         st.error("Blue directive is required")
        #         st.stop()

        # # Update the state
        # has_changed = False
        # if (
        #     "red_directive" in st.session_state
        #     and st.session_state["red_directive"] != red_directive
        # ) or "red_directive" not in st.session_state:
        #     has_changed = True
        #     st.session_state["red_directive"] = red_directive

        # if (
        #     "blue_directive" in st.session_state
        #     and st.session_state["blue_directive"] != blue_directive
        # ) or "blue_directive" not in st.session_state:
        #     has_changed = True
        #     st.session_state["blue_directive"] = blue_directive

        # if (
        #     "first_speaker" in st.session_state
        #     and st.session_state["first_speaker"] != speaker
        # ) or "first_speaker" not in st.session_state:
        #     has_changed = True
        #     st.session_state["first_speaker"] = speaker
        # has_changed = True

        # # if anything has changed, reset the session state
        # if has_changed:
        #     if "Red" in st.session_state:
        #         del st.session_state["Red"]
        #     if "Blue" in st.session_state:
        #         del st.session_state["Blue"]
        #     if "chat_history" in st.session_state:
        #         del st.session_state["chat_history"]
        #     if "response" in st.session_state:
        #         del st.session_state["response"]
        #     if "llm" in st.session_state:
        #         del st.session_state["llm"]
        #     if "memory" in st.session_state:
        #         del st.session_state["memory"]
        #     st.session_state["speaker"] = st.session_state["first_speaker"]
        #     st.success("Updated")
        # else:
        #     st.info("No changes")

# if not "red_directive" in st.session_state or not "blue_directive" in st.session_state:
#     st.stop()

#
# Session state
#
if "speaker" not in st.session_state:
    st.session_state["speaker"] = st.session_state["first_speaker"]

if "response" not in st.session_state:
    current_scenario_key = st.session_state["current_scenario_key"]
    current_scenario = scenario_library["scenarios"][current_scenario_key]
    st.session_state["response"] = current_scenario["prompt"]

if "tools" not in st.session_state:
    tools = load_tools(["ddg-search"])
    st.session_state["tools"] = tools

if "llm" not in st.session_state:
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.7,
        verbose=True,
    )
    st.session_state["llm"] = llm

if "memory" not in st.session_state:
    human_prefix = st.session_state["first_speaker"]
    ai_prefix = "Red" if human_prefix == "Blue" else "Blue"

    memory = SharedMemory(
        return_messages=True,
        human_prefix=human_prefix,
        ai_prefix=ai_prefix,
    )
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",
    #     return_messages=True,
    #     human_prefix=human_prefix,
    #     ai_prefix=ai_prefix,
    # )

    # memory = ConversationSummaryBufferMemory(
    #     memory_key="chat_history",
    #     return_messages=True,
    #     llm=st.session_state["llm"],
    #     max_token_limit=3000,
    #     human_prefix=human_prefix,
    #     ai_prefix=ai_prefix,
    # )
    st.session_state["memory"] = memory

    # we can use the creation of memory as a trigger to create
    # a new session
    session_file, session = session_create(
        scenario=st.session_state["current_scenario_key"],
        red_bot=st.session_state["red_bot"],
        red_directive=st.session_state["red_directive"],
        blue_bot=st.session_state["blue_bot"],
        blue_directive=st.session_state["blue_directive"],
        first_speaker=st.session_state["first_speaker"],
    )
    st.session_state["session_file"] = session_file
    st.session_state["session"] = session

    st.session_state["tokens"] = 0
    st.session_state["cost"] = 0


def create_agent(name: str) -> None:
    directive = (
        f"You are {name}. You are talking to {other_speaker(name)}. "
        + st.session_state[f"{name.lower()}_directive"]
    )
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
    st.session_state[name] = agent_chain


if "Red" not in st.session_state:
    create_agent("Red")

if "Blue" not in st.session_state:
    create_agent("Blue")


#
# Chat history
#

messages = st.session_state["memory"].chat_history.messages
for i in range(len(messages)):
    message = messages[i]

    if type(message) == HumanMessage:
        speaker = st.session_state["first_speaker"]
    else:
        speaker = other_speaker(st.session_state["first_speaker"])
    color = speaker_color(speaker)
    st.text_area(
        f":{color}[{speaker}]",
        value=message.content,
        key=f"chat_history_{i}",
    )


def add_message(speaker, message) -> None:
    chat_history = st.session_state["memory"].chat_history
    if speaker == st.session_state["first_speaker"]:
        chat_history.add_user_message(message)
    else:
        chat_history.add_ai_message(message)

    session_add_message(
        session_file=st.session_state["session_file"],
        session=st.session_state["session"],
        speaker=speaker,
        message=message,
    )


with st.form("chat_input"):
    speaker = st.session_state["speaker"]
    responder = other_speaker(speaker)
    responder_agent = st.session_state[responder]

    color = speaker_color(speaker)
    prompt = st.text_area(f":{color}[{speaker}]", value=st.session_state["response"])
    submit = st.form_submit_button("Send", use_container_width=True)
    with st.expander("Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Session Tokens: {st.session_state['tokens']}")
        with col2:
            st.text(f"Session Cost: {st.session_state['cost']:.3f}")
    if submit:
        add_message(speaker, prompt)
        with st.spinner("Thinking..."):
            st_callback = StreamlitCallbackHandler(st.container())
            with get_openai_callback() as cb:
                try:
                    response = responder_agent.run(
                        input=prompt, callbacks=[st_callback]
                    )

                except OutputParserException as e:
                    response = str(e)
                    prefix = "Could not parse LLM output: "
                    if not response.startswith(prefix):
                        raise e
                    response = response.removeprefix(prefix)

                print(cb)
                st.session_state["tokens"] += cb.total_tokens
                st.session_state["cost"] += cb.total_cost

            st.session_state["response"] = response
            st.session_state["speaker"] = responder

            st.experimental_rerun()
