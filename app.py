import os
from dotenv import load_dotenv, find_dotenv

from typing import Set
import streamlit as st
from streamlit_chat import message
from engn.engine import llm_call


def inpt_str(source_urls: Set[str]) -> str:
    _ = load_dotenv(find_dotenv())
    if not source_urls:
        return ""
    sources_list = sorted(source_urls)
    sources_str = "sources:\n"
    for i, source in enumerate(sources_list, start=1):
        sources_str += f"{i}. {source}\n"
    return sources_str


st.title(':blue["PDF Companion"] *- LLM Powered AI App* :smile:')

session_state = st.session_state

if "chat_answers_history" not in session_state:
    session_state["chat_answers_history"] = []
if "user_prompt_history" not in session_state:
    session_state["user_prompt_history"] = []
if "chat_history" not in session_state:
    session_state["chat_history"] = []

prompt = st.text_input("Prompt", placeholder="Enter the Query Related to the PDF Document Here...") or st.button(
    "Submit"
)

if prompt:
    with st.spinner("Finding Answers..."):
        generated_response = llm_call(
            query=prompt, chat_history=session_state["chat_history"]
        )

        source_documents = generated_response.get("source_documents", [])
        sources = set(doc.metadata["source"] for doc in source_documents)

        formatted_response = f"{generated_response['answer']} \n\n {inpt_str(sources)}"

        session_state.chat_history.append((prompt, generated_response["answer"]))
        session_state.user_prompt_history.append(prompt)
        session_state.chat_answers_history.append(formatted_response)

if session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        session_state["chat_answers_history"], session_state["user_prompt_history"]
    ):
        message(user_query, is_user=True)
        message(generated_response)
