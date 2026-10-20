
import asyncio
import hmac
import os
import threading

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agent import build_agent, DB
from security import safe_response  

load_dotenv()

st.set_page_config(
    page_title="Randoneo Support",
    page_icon="🏔️"
)

# Secrets locaux et Streamlit Cloud
for key in (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "APP_USER_1", "APP_PASSWORD_1", "APP_CLIENT_ID_1",
    "APP_USER_2", "APP_PASSWORD_2", "APP_CLIENT_ID_2"
):
    if key in st.secrets:
        os.environ[key] = str(st.secrets[key])


USERS = {
    os.getenv(f"APP_USER_{i}"): (
        os.getenv(f"APP_PASSWORD_{i}"),
        os.getenv(f"APP_CLIENT_ID_{i}")
    )
    for i in (1, 2)
    if os.getenv(f"APP_USER_{i}")
}


class AgentService:
    """Conserve l'agent sur une boucle asynchrone durable."""

    def __init__(self, client_id):
        self.client_id = client_id
        self.loop = asyncio.new_event_loop()

        threading.Thread(
            target=self._run_loop,
            daemon=True
        ).start()

        self.submit(self._initialize())

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coroutine):
        return asyncio.run_coroutine_threadsafe(
            coroutine, self.loop
        ).result()

    async def _initialize(self):
        self.context = AsyncSqliteSaver.from_conn_string(DB)
        self.checkpointer = await self.context.__aenter__()
        self.agent = await build_agent(
            self.checkpointer,
            customer_id="CUST-001"
        )

    async def _execute(self, payload, thread_id):
        return await self.agent.ainvoke(
            payload,
            config={"configurable": {"thread_id": thread_id}}
        )

    def ask(self, payload, thread_id):
        return self.submit(self._execute(payload, thread_id))


@st.cache_resource
def get_service(client_id):
    return AgentService(client_id)


def extract_answer(result):
    for message in reversed(result.get("messages", [])):
        if message.type == "ai" and isinstance(message.content, str):
            if message.content:
                return safe_response(message.content)
    return ""


def show_result(result):
    calls = [
        {"outil": call["name"], "arguments": call["args"]}
        for msg in result.get("messages", [])
        if msg.type == "ai"
        for call in getattr(msg, "tool_calls", [])
    ]

    if calls:
        with st.expander("Étapes de l'agent"):
            for call in calls:
                st.write(call["outil"])
                st.json(call["arguments"])

    if result.get("__interrupt__"):
        st.session_state.pending = result["__interrupt__"][0].value
        return

    answer = extract_answer(result)

    if answer:
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })


def login():
    st.title("🏔️ Randoneo Support")
    st.subheader("Connexion")

    with st.form("login"):
        username = st.text_input("Identifiant")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Se connecter")

    if submitted:
        user = USERS.get(username)

        if (
            user
            and user[0]
            and user[1]
            and hmac.compare_digest(password, user[0])
        ):
            st.session_state.client_id = user[1]
            st.session_state.messages = []
            st.session_state.pending = None
            st.rerun()

        st.error("Identifiants incorrects")


def main():
    if "client_id" not in st.session_state:
        login()
        st.stop()

    client_id = st.session_state.client_id
    thread_id = f"randoneo-{client_id}-main"
    service = get_service(client_id)

    st.title("🏔️ Randoneo Support")
    st.caption(f"Client : {client_id}")

    if st.button("Déconnexion"):
        for key in ("client_id", "messages", "pending"):
            st.session_state.pop(key, None)
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending" not in st.session_state:
        st.session_state.pending = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.pending:
        st.warning("Validation humaine requise")
        st.json(st.session_state.pending)

        col1, col2 = st.columns(2)

        approve = col1.button("Approuver")
        reject = col2.button("Refuser")

        if approve or reject:
            result = service.ask(
                Command(resume="approve" if approve else "reject"),
                thread_id
            )

            st.session_state.pending = None
            show_result(result)
            st.rerun()

        st.stop()

    question = st.chat_input("Posez votre question...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.spinner("Recherche en cours..."):
            result = service.ask(
                {
                    "messages": [
                        {"role": "user", "content": question}
                    ]
                },
                thread_id
            )

        show_result(result)
        st.rerun()


if __name__ == "__main__":
    main()
