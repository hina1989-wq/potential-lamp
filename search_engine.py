import os
import re
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS
import wikipedia
import arxiv


load_dotenv()

st.set_page_config(page_title="Mini ReAct Agent", page_icon="🤖")

st.title("🤖 Mini ReAct Agent Search Engine (Web + Wikipedia + arxiv)")

st.sidebar.header("Settings")

api_key = st.sidebar.text_input("Groq Api key",type ="password") or os.getenv("GROQ_API_KEY","")

model_name = st.sidebar.selectbox("Model",["llama-3.1-8b-instant","qwen/qwen3-32b"],index=0)
max_steps = st.sidebar.slider("Max reasoning steps",1,6,3)

st.markdown("""
     This is a simple agent app to make you understand how aget *thinks -> act -> observe -> conclude*
    using 3 Tools : **Duckduckgo**, **wikipedia** and **arxiv** 
""")

## Tool functions

def tool_web_search(query, k=4):
    with DDGS() as ddg:
        results = ddg.text(query, region="us-en", max_results=k)
        lines = []
        for r in results:
            title, link, body = r.get("title",""), r.get("href",""), r.get("body","")
            lines.append(f" - {title} - {link}\n {body}")
        return "\n".join(lines) if lines else "No Result Found."

def tool_wikipedia(query, sentences=2):
    try:
        wikipedia.set_lang("en")
        pages = wikipedia.search(query,results=1)
        if not pages:
            return "No Wikipedia page found."
        summary = wikipedia.summary(pages[0], sentences=sentences)
        return f"Wikipedia: {pages[0]}\n{(summary)}"
    except Exception as e:
        return f"Wikipedia error: {e}"
    
def tool_arxiv(query):
    try:
        search = arxiv.Search(query=query, max_results=1, sort_by=arxiv.SortCriterion.Relevance)
        results = list(search.results())
        if not results:
            return "No Arxiv paper found"
        paper = results[0]
        snippet = (paper.summary or "").replace("\n"," ")[:400]
        return f"arxiv: {paper.title}\n Link: {paper.entry_id}\n{snippet}..."
    except Exception as e:
        return f"arxiv error: {e}"
    
# ReActStyle Prompt

SYSTEM_PROMPT = """

You are a helpful research assistant with access to 3 tools:
1) Websearch 2) Wikipedia 3) Arxiv

Follow this reasoning format exactly:

Thought: what you will do next,
Action: which tool  to use (Websearch or Wikipedia or Arxiv),
Action Input: search phrase,
(Then you get an observation with the tool result.)

Repeat this loop untill can answer.
When read, write:
Final Answer: <your short, clear answer in English>

"""

ACTION_RE = re.compile(r"^Action:\s*(WebSearch|Wikipedia|Arxiv)",re.I)
INPUT_RE = re.compile(r"^Action Input:\s*(.*)",re.I)

## Simple React Agent loop
def mini_agent(client, model, question,max_iters=3):
    """Run a small reasoning loop manually (no langchain)."""
    transcript= [f"User Question: {question}"]
    observation = None 

    for step in range(1, max_iters + 1):
        # Build conversation seen by the model
        convo = SYSTEM_PROMPT+ "\n" + "\n".join(transcript)

        if observation:
            convo += f"\nObservation: {observation}"

        # Ask llm what to do next
        resp = client.chat.completions.create(
            model = model,
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":convo},
            ],
            temperature=0.2,
            max_tokens=912
        )

        text = resp.choices[0].message.content or ""

        with st.expander(f" Step {step}",expanded=False):
            st.write(text)

        if "Final Answer:" in text:
            return text.split("Final Answer:",1)[1].strip()
        
        action, action_input = None,None
        for line in text.splitlines():
            if ACTION_RE.match(line):
                action = ACTION_RE.match(line).group(1).title()
            if INPUT_RE.match(line):
                action_input = INPUT_RE.match(line).group(1).strip()
        
        if not action or not action_input:
            return "Could not understand next step."
        
        # Call the tool chosen by the model
        if action == "Websearch":
            observation = tool_web_search(action_input)
        elif action == "Wikipedia":
            observation = tool_wikipedia(action_input)
        elif action == "Arxiv":
            observation = tool_arxiv(action_input)
        else:
            observation =f"Unknown tool: {action}"

        # Record reasoning so far
        transcript.append(f"Thought: I will use {action}.")
        transcript.append(f"Action: {action}")
        transcript.append(f"Action Input: {action_input}")
        transcript.append(f"Observation: {observation}")
    
    # if max steps reached, ask model to summarize
    summary = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system", "content":"summarize briefly in english."},
            {"role":"user","content": "\n".join(transcript)},
        ],
        temperature=0.2,
        max_tokens=656,
    )
    return summary.choices[0].message.content

# streamlit ui - ask a question

query = st.chat_input("Ask me anything....")

if query:
    st.chat_message("user").write(query)

    if not api_key:
        st.error("Please add your GROQ_API_KEY in sidebar or .env")
    else:
        client = Groq(api_key=api_key)
        with st.spinner("Thinking..."):
            answer = mini_agent(client,model=model_name,question=query, max_iters=max_steps)
        st.chat_message("assistant").write(answer)

