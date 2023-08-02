from datetime import datetime
import json
import os
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain.agents import ConversationalChatAgent, load_tools, AgentExecutor
from langchain.callbacks import StreamlitCallbackHandler, get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ChatMessageHistory
from langchain.schema import (
    BaseMemory,
    HumanMessage,
    OutputParserException,
)
from langchain.schema.language_model import BaseLanguageModel

import streamlit as st
from streamlit_chat import message

#
# Load environment variables
#

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATA_DIR = os.getenv("DATA_DIR", "./data")


#
# Helpers
#


def generate_string_from_messages(messages: list[dict[str, str]]) -> str:
    message_string = ""

    for message in messages:
        speaker = message.get("speaker")
        message_text = message.get("message")
        message_string += f"### {speaker}: {message_text}"

    return message_string


#
# Custom memory that is shared between two agents
#


class Message(BaseModel):
    speaker: str
    message: str

    def __str__(self) -> str:
        return f"### {speaker}: {message}"


class Session(BaseModel):
    scenario: str
    red_bot: str
    red_directive: str
    blue_bot: str
    blue_directive: str
    first_speaker: str
    tokens: int = 0
    cost: float = 0.0
    messages: list[Message] = []


class SessionMemory(BaseMemory):
    return_messages: bool = True

    chat_history: ChatMessageHistory = ChatMessageHistory()
    memory_key: str = "chat_history"
    llm: BaseLanguageModel = None
    session_file: str = None
    session: Session = None
    transcript: str = ""

    def __init__(
        self,
        human_prefix: str,
        ai_prefix: str,
        scenario: str,
        red_bot: str,
        red_directive: str,
        blue_bot: str,
        blue_directive: str,
        first_speaker: str,
    ) -> None:
        super().__init__(
            human_prefix=human_prefix,
            ai_prefix=ai_prefix,
        )
        self.llm = OpenAI()
        self.session = Session(
            scenario=scenario,
            red_bot=red_bot,
            red_directive=red_directive,
            blue_bot=blue_bot,
            blue_directive=blue_directive,
            first_speaker=first_speaker,
        )
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_file = os.path.join(DATA_DIR, f"{scenario} {timestamp}.json")

    #
    # Boilerplate
    #

    def clear(self) -> None:
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
    # Sessions
    #

    def session_write(
        self,
    ) -> None:
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        with open(self.session_file, "w") as f:
            json.dump(self.session.dict(), f, indent=2)

    def add_message(self, speaker, message) -> None:
        m = Message(speaker=speaker, message=message)
        self.transcript += str(m)

        if speaker == st.session_state["first_speaker"]:
            self.chat_history.add_user_message(message)
        else:
            self.chat_history.add_ai_message(message)

        self.session.messages.append(m)
        self.session_write()

    def session_close(self) -> None:
        if len(self.session.messages) > 1:
            self.session_write()
        else:
            os.remove(self.session_file)


#
# Scenario library
#


def scenario_library_load() -> dict[str, any]:
    with open("scenario_library.json", "r") as f:
        scenario_library = json.load(f)
    return scenario_library


def scenario_library_save(scenario_library: dict[str, any]) -> None:
    with open("scenario_library.json", "w") as f:
        json.dump(scenario_library, f, indent=2)


def scenario_library_from_session_state() -> None:
    scenario_library = st.session_state["scenario_library"]
    current_scenario_key = st.session_state["current_scenario_key"]
    scenario_library["current_scenario_key"] = current_scenario_key
    scenario = scenario_library["scenarios"][current_scenario_key]
    scenario["first_speaker"] = st.session_state["first_speaker"]
    scenario["prompt"] = st.session_state["prompt"]
    red_bot = st.session_state["red_bot"]
    blue_bot = st.session_state["blue_bot"]
    scenario["Red"] = red_bot
    scenario["Blue"] = blue_bot
    scenario_library["bots"][red_bot] = st.session_state["red_directive"]
    scenario_library["bots"][blue_bot] = st.session_state["blue_directive"]


def scenario_library_to_session_state() -> None:
    scenario_library = st.session_state["scenario_library"]
    current_scenario_key = scenario_library["current_scenario_key"]
    if current_scenario_key:
        scenario = scenario_library["scenarios"][current_scenario_key]
        st.session_state["first_speaker"] = scenario["first_speaker"]
        st.session_state["prompt"] = scenario["prompt"]
        st.session_state["red_bot"] = scenario["Red"]
        st.session_state["blue_bot"] = scenario["Blue"]
        bots = scenario_library["bots"]
        red_bot = scenario["Red"]
        blue_bot = scenario["Blue"]
        st.session_state["red_directive"] = bots[red_bot] if red_bot in bots else ""
        st.session_state["blue_directive"] = bots[blue_bot] if blue_bot in bots else ""
    else:
        st.session_state["first_speaker"] = "Red"
        st.session_state["prompt"] = ""
        st.session_state["red_bot"] = ""
        st.session_state["blue_bot"] = ""
        st.session_state["red_directive"] = ""
        st.session_state["blue_directive"] = ""


if "scenario_library" not in st.session_state:
    st.session_state["scenario_library"] = scenario_library_load()

#
# Streamlit app
#

st.set_page_config(
    page_title="Red v Blue",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
<div class="header-container">
  <img src="./app/static/red-v-blue.png" alt="Red v Blue" class="header-image">
  <div class="header-text">
    <h2><span class="red-word">Red</span> v <span class="blue-word">Blue</span></h2>
    <p>Human-mediated AI agent-to-agent chat</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

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


def reset_dialog() -> None:
    # commit the final response, if possible
    if (
        "speaker" in st.session_state
        and "response" in st.session_state
        and "memory" in st.session_state
    ):
        speaker = st.session_state["speaker"]
        response = st.session_state["response"]
        memory = st.session_state["memory"]
        memory.add_message(speaker, response)

    # clean up session state
    if "speaker" in st.session_state:
        del st.session_state["speaker"]
    if "Red" in st.session_state:
        del st.session_state["Red"]
    if "Blue" in st.session_state:
        del st.session_state["Blue"]
    if "response" in st.session_state:
        del st.session_state["response"]
    if "memory" in st.session_state:
        memory = st.session_state["memory"]
        memory.session_close()
        del st.session_state["memory"]


if "current_scenario_key" not in st.session_state:
    scenario_library_to_session_state()

with st.expander("Directives", expanded=True):
    st.markdown("#### Scenario")

    scenario_library = st.session_state["scenario_library"]
    scenarios = list(scenario_library["scenarios"])
    bots = list(scenario_library["bots"])

    def on_scenario_changed() -> None:
        current_scenario_key = st.session_state["current_scenario_key"]
        scenario_library["current_scenario_key"] = current_scenario_key
        scenario_library_to_session_state()
        reset_dialog()

    st.selectbox(
        "Scenario",
        scenarios,
        label_visibility="collapsed",
        key="current_scenario_key",
        on_change=on_scenario_changed,
    )

    col1, col2 = st.columns(2)
    with col1:
        new_scenario = st.button(
            "New Scenario",
            use_container_width=True,
        )
    with col2:
        delete_scenario = st.button(
            "Delete Scenario",
            use_container_width=True,
            disabled=len(scenarios) == 0,
        )
    if new_scenario:
        print("new_scenario")
        with st.form(key="new_scenario"):
            st.text_input(
                "New scenario",
                placeholder="Scenario name",
                label_visibility="collapsed",
                key="new_scenario_name",
            )

            def on_create() -> None:
                new_scenario_name = st.session_state["new_scenario_name"]
                if len(new_scenario_name) == 0:
                    st.warning(
                        "Scenario name cannot be empty",
                        icon="⚠️",
                    )
                    return
                bots = list(scenario_library["bots"].keys())
                bot = "" if not bots else bots[0]
                scenario_library["scenarios"][new_scenario_name] = {
                    "first_speaker": "Red",
                    "Red": bot,
                    "Blue": bot,
                    "prompt": "",
                }
                st.session_state["current_scenario_key"] = scenario_library[
                    "current_scenario_key"
                ] = new_scenario_name
                scenario_library_save(scenario_library)
                scenario_library_to_session_state()
                reset_dialog()

            create_new_scenario = st.form_submit_button(
                "Create",
                type="primary",
                use_container_width=True,
                on_click=on_create,
            )

    if delete_scenario:

        def on_delete_scenario() -> None:
            current_scenario_key = st.session_state["current_scenario_key"]
            scenario_library["scenarios"].pop(current_scenario_key)
            scenarios = list(scenario_library["scenarios"].keys())
            replace_scenario = "" if not scenarios else scenarios[0]
            scenario_library["current_scenario_key"] = replace_scenario
            scenario_library_save(scenario_library)
            scenario_library_to_session_state()
            reset_dialog()

        with st.form(key="delete_scenario"):
            current_scenario_key = st.session_state["current_scenario_key"]
            st.warning(
                f"Are you sure you want to delete {current_scenario_key}?",
                icon="⚠️",
            )
            st.form_submit_button(
                "Delete",
                type="primary",
                use_container_width=True,
                on_click=on_delete_scenario,
            )

    with st.container():
        st.markdown("#### Bots")
        col1, col2 = st.columns(2)

        def on_bot_changed(bot_key: str) -> None:
            bot_lower = bot_key.lower()
            current_scenario_key = st.session_state["current_scenario_key"]
            if not current_scenario_key:
                return
            scenario = scenario_library["scenarios"][current_scenario_key]
            scenario[bot_key] = st.session_state[f"{bot_lower}_bot"]
            scenario_library_save(scenario_library)
            scenario_library_to_session_state()
            reset_dialog()

        def on_directive_changed(bot_key: str) -> None:
            bot_lower = bot_key.lower()
            bot_directive = st.session_state[f"{bot_lower}_directive"]
            bot = st.session_state[f"{bot_lower}_bot"]
            scenario_library["bots"][bot] = bot_directive
            scenario_library_save(scenario_library)
            scenario_library_to_session_state()
            reset_dialog()

        # there is a special case where there are no scenarios, but
        # there are bots and therefore None or "" is not a valid option
        # for the selectbox. In this case, we just set the bot to the
        # first bot in the list
        if len(bots) > 0:
            if (
                "red_bot" not in st.session_state
                or st.session_state["red_bot"] not in bots
            ):
                st.session_state["red_bot"] = bots[0]
            if (
                "blue_bot" not in st.session_state
                or st.session_state["blue_bot"] not in bots
            ):
                st.session_state["blue_bot"] = bots[0]

            # ensure the directives match the bots
            st.session_state["red_directive"] = scenario_library["bots"][
                st.session_state["red_bot"]
            ]
            st.session_state["blue_directive"] = scenario_library["bots"][
                st.session_state["blue_bot"]
            ]

        with col1:
            st.selectbox(
                ":red[🤖 Red Bot]",
                bots,
                key="red_bot",
                on_change=lambda: on_bot_changed("Red"),
            )
            st.text_area(
                ":red[🤖 Red Directive]",
                key="red_directive",
                on_change=lambda: on_directive_changed("Red"),
            )
            delete_red_bot = st.button(
                "Delete Red Bot",
                use_container_width=True,
                disabled=len(bots) == 0,
            )

        with col2:
            st.selectbox(
                ":blue[🤖 Blue Bot]",
                bots,
                key="blue_bot",
                on_change=lambda: on_bot_changed("Blue"),
            )
            st.text_area(
                ":blue[🤖 Blue Directive]",
                key="blue_directive",
                on_change=lambda: on_directive_changed("Blue"),
            )
            delete_blue_bot = st.button(
                "Delete Blue Bot",
                use_container_width=True,
                disabled=len(bots) == 0,
            )

        if delete_red_bot or delete_blue_bot:

            def delete_bot(bot: str) -> None:
                del scenario_library["bots"][bot]
                bots = list(scenario_library["bots"].keys())
                replace_bot = "" if not bots else bots[0]
                for scenario in scenario_library["scenarios"].values():
                    if scenario["Red"] == bot:
                        scenario["Red"] = replace_bot
                    if scenario["Blue"] == bot:
                        scenario["Blue"] = replace_bot
                scenario_library_save(scenario_library)
                scenario_library_to_session_state()
                reset_dialog()

            with st.form(key="delete_bot"):
                bot_key = "red_bot" if delete_red_bot else "blue_bot"
                bot_name = st.session_state[bot_key]
                st.warning(
                    f"Are you sure you want to delete {bot_name}?",
                    icon="⚠️",
                )
                st.form_submit_button(
                    "Delete",
                    type="primary",
                    use_container_width=True,
                    on_click=lambda: delete_bot(bot_name),
                )

        new_bot = st.button("New Bot", use_container_width=True)
        if new_bot:

            def on_create() -> None:
                new_bot_name = st.session_state["new_bot_name"]
                if len(new_bot_name) == 0:
                    st.warning(
                        "Bot name cannot be empty",
                        icon="⚠️",
                    )
                    return
                scenario_library["bots"][new_bot_name] = ""
                bots = list(scenario_library["bots"].keys())
                replace_bot = "" if not bots else bots[0]
                for scenario in scenario_library["scenarios"].values():
                    if not scenario["Red"] or scenario["Red"] == "":
                        scenario["Red"] = replace_bot
                    if not scenario["Blue"] or scenario["Blue"] == "":
                        scenario["Blue"] = replace_bot
                print(scenario_library)
                scenario_library_save(scenario_library)
                scenario_library_to_session_state()
                reset_dialog()

            with st.form(key="new_bot"):
                st.text_input(
                    "New bot",
                    key="new_bot_name",
                    placeholder="Bot name",
                    label_visibility="collapsed",
                )
                st.form_submit_button(
                    "Create",
                    type="primary",
                    use_container_width=True,
                    on_click=on_create,
                )

        def on_first_speaker_changed() -> None:
            current_scenario_key = st.session_state["current_scenario_key"]
            scenario = scenario_library["scenarios"][current_scenario_key]
            scenario["first_speaker"] = st.session_state["first_speaker"]
            scenario_library_save(scenario_library)
            scenario_library_to_session_state()
            reset_dialog()

        st.selectbox(
            "First speaker",
            speakers,
            key="first_speaker",
            on_change=on_first_speaker_changed,
        )

        def on_prompt_changed() -> None:
            current_scenario_key = st.session_state["current_scenario_key"]
            scenario = scenario_library["scenarios"][current_scenario_key]
            scenario["prompt"] = st.session_state["prompt"]
            scenario_library_save(scenario_library)
            scenario_library_to_session_state()
            reset_dialog()

        st.text_area(
            "Prompt",
            key=f"prompt",
            on_change=on_prompt_changed,
        )

        st.button(
            "Reset Dialog",
            type="primary",
            use_container_width=True,
            on_click=reset_dialog,
        )

#
# Validate the configuration before continuing
#
if not st.session_state["red_bot"] or not st.session_state["blue_bot"]:
    st.stop()

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

    memory = SessionMemory(
        human_prefix=human_prefix,
        ai_prefix=ai_prefix,
        scenario=st.session_state["current_scenario_key"],
        red_bot=st.session_state["red_bot"],
        red_directive=st.session_state["red_directive"],
        blue_bot=st.session_state["blue_bot"],
        blue_directive=st.session_state["blue_directive"],
        first_speaker=st.session_state["first_speaker"],
    )
    st.session_state["memory"] = memory
    scenario_library_save(st.session_state["scenario_library"])


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


with st.form("chat_input"):
    speaker = st.session_state["speaker"]
    responder = other_speaker(speaker)
    responder_agent = st.session_state[responder]

    color = speaker_color(speaker)
    prompt = st.text_area(f":{color}[{speaker}]", value=st.session_state["response"])
    submit = st.form_submit_button("Send", use_container_width=True)

    memory = st.session_state["memory"]
    session = memory.session
    with st.expander("Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Session Tokens: {session.tokens}")
        with col2:
            st.text(f"Session Cost: {session.cost:.3f}")
    if submit:
        memory.add_message(speaker, prompt)
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

                session.tokens += cb.total_tokens
                session.cost += cb.total_cost

            st.session_state["response"] = response
            st.session_state["speaker"] = responder

            st.experimental_rerun()
