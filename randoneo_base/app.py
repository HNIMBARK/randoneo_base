
import asyncio
import json
import threading

import streamlit as st

from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import build_agent, DB


st.set_page_config(
    page_title="Randoneo Support",
    page_icon="🏔️"
)

# Identifiants fictifs : démonstration uniquement.
USERS = {
    "client1": ("demo1", "CLIENT-001"),
    "client2": ("demo2", "CLIENT-002")
}


class AgentService:
    """Agent et checkpointer sur une boucle durable."""

    def __init__(self, customer_id):
        self.customer_id = customer_id
        self.loop = asyncio.new_event_loop()

        self.thread = threading.Thread(
            target=self._run_loop,
            daemon=True
        )
        self.thread.start()

        self.submit(self._initialize())

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coroutine):
        future = asyncio.run_coroutine_threadsafe(
            coroutine,
            self.loop
        )
        return future.result()

    async def _initialize(self):
        self.context = (
            AsyncSqliteSaver.from_conn_string(DB)
        )
        self.checkpointer = await self.context.__aenter__()

        self.agent = await build_agent(
            self.checkpointer,
        )

    async def _execute(self, payload, thread_id):
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        return await self.agent.ainvoke(
            payload,
            config=config
        )

    def ask(self, payload, thread_id):
        return self.submit(
            self._execute(payload, thread_id)
        )


@st.cache_resource
def get_service(customer_id):
    return AgentService(customer_id)


def extract_response(result):
    for message in reversed(result.get("messages", [])):
        if (
            message.type == "ai"
            and isinstance(message.content, str)
            and message.content
        ):
            return message.content
    return ""


def extract_steps(result):
    steps = []

    for message in result.get("messages", []):
        if message.type == "ai":
            for call in getattr(message, "tool_calls", []):
                steps.append({
                    "tool": call["name"],
                    "arguments": call["args"]
                })

    return steps


def login():
    st.title("🏔️ Randoneo Support")
    st.subheader("Connexion")

    with st.form("login"):
        username = st.text_input("Identifiant")
        password = st.text_input(
            "Mot de passe",
            type="password"
        )
        submitted = st.form_submit_button(
            "Se connecter"
        )

    if submitted:
        user = USERS.get(username)

        if user and password == user[0]:
            st.session_state.customer_id = user[1]
            st.session_state.messages = []
            st.session_state.pending = None
            st.rerun()

        st.error("Identifiants incorrects")


def show_result(result):
    steps = extract_steps(result)

    if steps:
        with st.expander(
            "Étapes et outils de l'agent",
            expanded=True
        ):
            for step in steps:
                st.write("Outil :", step["tool"])
                st.json(step["arguments"])

    if result.get("__interrupt__"):
        pending = result["__interrupt__"][0].value

        st.session_state.pending = pending
        st.warning("Action en attente de validation")

        # Affiche les arguments réels.
        st.json(pending)
        return

    answer = extract_response(result)

    if answer:
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message("assistant"):
            st.write(answer)


def main():
    if "customer_id" not in st.session_state:
        login()
        st.stop()

    customer_id = st.session_state.customer_id
    thread_id = f"randoneo-{customer_id}-main"

    service = get_service(customer_id)

    st.title("🏔️ Randoneo Support")
    st.caption(f"Client connecté : {customer_id}")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending" not in st.session_state:
        st.session_state.pending = None

    if st.button("Déconnexion"):
        for key in (
            "customer_id",
            "messages",
            "pending"
        ):
            st.session_state.pop(key, None)
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.pending is not None:
        st.warning("Confirmez l'action proposée")
        st.json(st.session_state.pending)

        col1, col2 = st.columns(2)

        with col1:
            approve = st.button("Approuver")

        with col2:
            reject = st.button("Refuser")

        if approve or reject:
            decision = (
                "approve" if approve else "reject"
            )

            result = service.ask(
                Command(resume=decision),
                thread_id
            )

            st.session_state.pending = None
            show_result(result)
            st.rerun()

        st.stop()

    question = st.chat_input(
        "Posez votre question..."
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.spinner("L'agent analyse la demande..."):
            result = service.ask(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                },
                thread_id
            )

        show_result(result)
        st.rerun()


if __name__ == "__main__":
    main()
