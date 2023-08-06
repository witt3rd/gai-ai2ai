from datetime import datetime
import json
import os
from typing import Literal

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

from config import get_config
from session_model import (
    Message,
    Session,
    session_dir,
    session_load,
    session_purge,
    session_rename,
    session_save,
    session_transcript_save,
    session_transcript_load,
)


#
# Helpers
#

speakers = ["Red", "Blue"]


def speaker_color(speaker) -> Literal["red", "blue"]:
    color = "red" if speaker == "Red" else "blue"
    return color


def other_speaker(speaker) -> Literal["Red", "Blue"]:
    other = "Red" if speaker == "Blue" else "Blue"
    return other


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


class SessionMemory(BaseMemory):
    return_messages: bool = True

    chat_history: ChatMessageHistory = ChatMessageHistory()
    memory_key: str = "chat_history"
    llm: BaseLanguageModel = None
    session: Session = None
    transcript: str = ""

    # can be initialized with a session or with a scenario and bots
    def __init__(
        self,
        session: Session,
    ) -> None:
        human_prefix = session.first_speaker
        ai_prefix = "Red" if human_prefix == "Blue" else "Blue"

        super().__init__(
            human_prefix=human_prefix,
            ai_prefix=ai_prefix,
        )
        self.session = session
        for message in session.messages:
            self.add_message_to_chat_history(message)
        self.llm = OpenAI(openai_api_key=get_config().OPENAI_API_KEY)

    #
    # Boilerplate
    #

    def clear(self) -> None:
        self.chat_history = ChatMessageHistory()

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, any]) -> dict[str, str]:
        return {self.memory_key: self.chat_history.messages}

    def save_context(self, inputs: dict[str, any], outputs: dict[str, str]) -> None:
        pass

    #
    # Sessions
    #

    def add_message_to_chat_history(self, msg: Message) -> None:
        if msg.speaker == self.session.first_speaker:
            self.chat_history.add_user_message(msg.message)
        else:
            self.chat_history.add_ai_message(msg.message)

    def add_message(self, speaker, message) -> None:
        m = Message(speaker=speaker, message=message)
        self.transcript += str(m)
        self.add_message_to_chat_history(m)
        self.session.messages.append(m)
        session_save(self.session, get_config().SESSION_DIR)

    def session_close(self) -> None:
        if len(self.session.messages) > 1:
            session_save(self.session, get_config().SESSION_DIR)
        # session_purge(get_config().SESSION_DIR, self.session)


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
    if not current_scenario_key or len(current_scenario_key) == 0:
        scenarios = list(scenario_library["scenarios"].keys())
        if len(scenarios) > 0:
            current_scenario_key = scenarios[0]
            scenario_library["current_scenario_key"] = current_scenario_key
        else:
            current_scenario_key = None

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
        st.session_state["current_scenario_key"] = current_scenario_key
    else:
        st.session_state["first_speaker"] = "Red"
        st.session_state["prompt"] = ""
        st.session_state["red_bot"] = ""
        st.session_state["blue_bot"] = ""
        st.session_state["red_directive"] = ""
        st.session_state["blue_directive"] = ""
        st.session_state["current_scenario_key"] = ""


def scenario_library_from_session(session: Session) -> None:
    if not session:
        return
    st.session_state["session"] = session
    scenario_library = st.session_state["scenario_library"]
    current_scenario_key = session.scenario
    scenario_library["current_scenario_key"] = st.session_state[
        "current_scenario_key"
    ] = current_scenario_key
    scenario = {
        "first_speaker": session.first_speaker,
        "prompt": session.prompt,
        "Red": session.red_bot,
        "Blue": session.blue_bot,
    }
    scenario_library["scenarios"][current_scenario_key] = scenario

    def update_bot_scenario(bot_color, directive, scenario_library) -> None:
        if bot_color not in scenario_library["bots"]:
            scenario_library["bots"][bot_color] = directive
        else:
            bot_directive = scenario_library["bots"][bot_color]
            if bot_directive != directive:
                scenario_library["bots"][bot_color] = directive

    update_bot_scenario(session.red_bot, session.red_directive, scenario_library)
    update_bot_scenario(session.blue_bot, session.blue_directive, scenario_library)


def session_from_session_state(
    name: str | None = None,
    timestamp: str | None = None,
) -> Session:
    if not timestamp:
        timestamp = datetime.now().strftime(get_config().DATETIME_FORMAT)

    if not name:
        scenario = (
            st.session_state["current_scenario_key"]
            if "current_scenario_key" in st.session_state
            else "New Session"
        )
        name = f"{scenario} {timestamp}"

    session = Session(
        name=name,
        timestamp=timestamp,
        scenario=st.session_state["current_scenario_key"],
        red_bot=st.session_state["red_bot"],
        red_directive=st.session_state["red_directive"],
        blue_bot=st.session_state["blue_bot"],
        blue_directive=st.session_state["blue_directive"],
        first_speaker=st.session_state["first_speaker"],
        prompt=st.session_state["prompt"],
    )
    return session


def reset_dialog() -> None:
    # commit the final response, if possible
    if (
        "speaker" in st.session_state
        and "response" in st.session_state
        and "memory" in st.session_state
    ):
        speaker = st.session_state["speaker"]
        response = st.session_state["response"]
        prompt = st.session_state["prompt"]
        if response != prompt:
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


def on_new_session() -> None:
    session = session_from_session_state()
    session_save(session, get_config().SESSION_DIR)
    st.session_state["session"] = session
    st.session_state["current_session_name"] = session.name


def on_rename_session() -> None:
    session = st.session_state["session"]
    name = st.session_state["new_session_name"]
    if len(name) == 0:
        st.warning(
            "Session name cannot be empty",
            icon="⚠️",
        )
        return
    session_rename(
        session,
        get_config().SESSION_DIR,
        name,
    )
    session.name = name
    st.session_state["current_session_name"] = session.name


#
# Session state
#

if "scenario_library" not in st.session_state:
    st.session_state["scenario_library"] = scenario_library_load()

if "current_scenario_key" not in st.session_state:
    scenario_library_to_session_state()

if "session" not in st.session_state:
    session = session_from_session_state()
    session_save(session, get_config().SESSION_DIR)
    st.session_state["session"] = session

if "current_session_name" not in st.session_state:
    st.session_state["current_session_name"] = session.name

#
# Streamlit app
#

st.set_page_config(
    page_title="InsightCrafter™",
    page_icon="💡",
)

if get_config().OPENAI_API_KEY is None:
    st.error(
        "OpenAI API key not found. Please set OPENAI_API_KEY environment variable.",
        icon="🔑",
    )
    st.stop()


#
# UI
#

# Custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
<div class="header-container">
<img src="./app/static/hero.png" alt="InsightCrafter" class="header-image">
<div class="header-text">
    <h2>InsightCrafter™</h2>
    <p>Crafting Clarity through Discourse</p>
</div>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Main", "About"])

with tab1:
    with st.expander("Sessions", expanded=True):
        session_list = session_dir(get_config().SESSION_DIR)
        session = st.session_state["session"]

        def on_session_changed() -> None:
            reset_dialog()
            current_session_name = st.session_state["current_session_name"]
            session = session_load(current_session_name, get_config().SESSION_DIR)
            scenario_library_from_session(session)
            scenario_library_to_session_state()

        st.selectbox(
            "Sessions",
            session_list,
            label_visibility="collapsed",
            key="current_session_name",
            on_change=on_session_changed,
        )

        tokens = session.tokens if session else 0
        cost = session.cost if session else 0.0
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Timestamp: {session.timestamp}")
            st.text(f"Session Tokens: {tokens}")
        with col2:
            st.text(f"Messages: {len(session.messages)}")
            st.text(f"Session Cost: {cost:.3f}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            new_session = st.button(
                "New Session",
                use_container_width=True,
                on_click=on_new_session,
            )
        with col2:
            rename_session = st.button(
                "Rename Session",
                use_container_width=True,
            )
        with col3:
            delete_session = st.button(
                "Delete Session",
                use_container_width=True,
                disabled=True,
            )
        with col4:
            st.button(
                "Cleanup Sessions",
                use_container_width=True,
                on_click=lambda: session_purge(get_config().SESSION_DIR, session),
            )

        if rename_session:
            with st.form(key="rename_session_form"):
                st.session_state["new_session_name"] = session.name
                st.text_input(
                    "Name",
                    placeholder="Session name",
                    label_visibility="collapsed",
                    key="new_session_name",
                )

                st.form_submit_button(
                    "Rename",
                    type="primary",
                    use_container_width=True,
                    on_click=on_rename_session,
                )

    with st.expander("Directives", expanded=True):
        st.markdown("#### Scenario")

        print(f"Current scenario: {st.session_state['current_scenario_key']}")
        scenario_library = st.session_state["scenario_library"]
        scenarios = list(scenario_library["scenarios"])
        bots = list(scenario_library["bots"])

        def on_scenario_changed() -> None:
            reset_dialog()
            current_scenario_key = st.session_state["current_scenario_key"]
            scenario_library["current_scenario_key"] = current_scenario_key
            scenario_library_to_session_state()

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
            with st.form(key="new_scenario"):
                st.text_input(
                    "New scenario",
                    placeholder="Scenario name",
                    label_visibility="collapsed",
                    key="new_scenario_name",
                )

                def on_create() -> None:
                    reset_dialog()
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

                create_new_scenario = st.form_submit_button(
                    "Create",
                    type="primary",
                    use_container_width=True,
                    on_click=on_create,
                )

        if delete_scenario:

            def on_delete_scenario() -> None:
                reset_dialog()
                current_scenario_key = st.session_state["current_scenario_key"]
                scenario_library["scenarios"].pop(current_scenario_key)
                scenarios = list(scenario_library["scenarios"].keys())
                replace_scenario = "" if not scenarios else scenarios[0]
                scenario_library["current_scenario_key"] = replace_scenario
                scenario_library_save(scenario_library)
                scenario_library_to_session_state()

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
                reset_dialog()
                bot_lower = bot_key.lower()
                current_scenario_key = st.session_state["current_scenario_key"]
                if not current_scenario_key:
                    return
                scenario = scenario_library["scenarios"][current_scenario_key]
                scenario[bot_key] = st.session_state[f"{bot_lower}_bot"]
                scenario_library_save(scenario_library)
                scenario_library_to_session_state()

            def on_directive_changed(bot_key: str) -> None:
                reset_dialog()
                bot_lower = bot_key.lower()
                bot_directive = st.session_state[f"{bot_lower}_directive"]
                bot = st.session_state[f"{bot_lower}_bot"]
                scenario_library["bots"][bot] = bot_directive
                scenario_library_save(scenario_library)
                scenario_library_to_session_state()

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

            with col2:
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

            if delete_red_bot or delete_blue_bot:

                def delete_bot(bot: str) -> None:
                    reset_dialog()
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
                    reset_dialog()
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
                    scenario_library_save(scenario_library)
                    scenario_library_to_session_state()

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
                reset_dialog()
                current_scenario_key = st.session_state["current_scenario_key"]
                scenario = scenario_library["scenarios"][current_scenario_key]
                scenario["first_speaker"] = st.session_state["first_speaker"]
                scenario_library_save(scenario_library)
                scenario_library_to_session_state()

            st.selectbox(
                "First speaker",
                speakers,
                key="first_speaker",
                on_change=on_first_speaker_changed,
            )

            def on_prompt_changed() -> None:
                reset_dialog()
                current_scenario_key = st.session_state["current_scenario_key"]
                scenario = scenario_library["scenarios"][current_scenario_key]
                scenario["prompt"] = st.session_state["prompt"]
                scenario_library_save(scenario_library)
                scenario_library_to_session_state()

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
            openai_api_key=get_config().OPENAI_API_KEY,
        )
        st.session_state["llm"] = llm

    if "memory" not in st.session_state:
        memory = (
            SessionMemory(
                scenario=st.session_state["current_scenario_key"],
                red_bot=st.session_state["red_bot"],
                red_directive=st.session_state["red_directive"],
                blue_bot=st.session_state["blue_bot"],
                blue_directive=st.session_state["blue_directive"],
                first_speaker=st.session_state["first_speaker"],
            )
            if "session" not in st.session_state
            else SessionMemory(session=st.session_state["session"])
        )
        st.session_state["memory"] = memory

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
    if len(messages) > 1:
        with st.expander("Chat History", expanded=True):
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

            if len(messages) > 0:
                st.button(
                    "Clear Chat History",
                    use_container_width=True,
                    on_click=st.session_state["memory"].clear,
                )

        with st.expander("Transcript", expanded=False):
            st.button(
                "Generate Transcript",
                use_container_width=True,
                on_click=lambda: session_transcript_save(
                    st.session_state["session"],
                    get_config().TRANSCRIPT_DIR,
                ),
            )

            session = st.session_state["session"]
            transcript = session_transcript_load(
                get_config().TRANSCRIPT_DIR,
                session.name,
                session.timestamp,
            )
            if transcript:
                st.markdown(transcript, unsafe_allow_html=True)

    with st.form("chat_input"):
        speaker = st.session_state["speaker"]
        responder = other_speaker(speaker)
        responder_agent = st.session_state[responder]

        color = speaker_color(speaker)
        prompt = st.text_area(
            f":{color}[{speaker}]", value=st.session_state["response"]
        )
        submit = st.form_submit_button("Send", use_container_width=True)

        memory = st.session_state["memory"]
        session = memory.session
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

with tab2:
    with open("README.md", "r") as f:
        readme = f.read()
    readme = readme.replace("static/", "./app/static/")
    st.markdown(readme, unsafe_allow_html=True)
