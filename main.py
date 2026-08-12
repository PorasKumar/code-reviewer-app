import base64
import streamlit as st

from pathlib import Path
import uuid
import time

from typing import Any,Optional
import os 
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

from src.graph import code_review_ai_app

load_dotenv()

st.set_page_config(
    page_title="AI Code Review Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

##############################
# 2. INLINE CSS & BACKGROUND #
##############################

def load_css(file_path: str):
    css_path = Path(file_path)
    if css_path.is_file():
        return css_path.read_text(encoding="utf-8")
    return ""

inline_css = load_css("src/inline_css_code.css")
if inline_css:
    st.markdown(f"<style>{inline_css}</style>", unsafe_allow_html=True)

#image
IMAGE_PATH = r"C:\Artificial Intelligence\Projects\Code Reviewer AI\AI Code Reviewer\background.png"

try:
    with open(IMAGE_PATH, "rb") as img_file:
        bg_style = f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"
    st.markdown(
        f"""
        <style>
        .stApp, div[data-testid="stAppViewContainer"] {{
            background-image: url('{bg_style}') !important;
            background-size: cover !important;
            background-position: center !important;
            background-repeat: no-repeat !important;
            background-attachment: fixed !important;
        }}
        header[data-testid="stHeader"] {{
            background: transparent !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    pass



######################
#  Session Variables #
######################
if "review_button" not in st.session_state:
    st.session_state.review_button = False

if "submitted_url" not in st.session_state:
    st.session_state.submitted_url = ""

if "userid" not in st.session_state:
    import uuid
    st.session_state.userid = str(f"user_{uuid.uuid4().hex[:10]}_{int(time.time())}")



###########################
# Cutom Loading Animation #
###########################
def custom_loader(message: str):
    return st.markdown(
        f"""
    <div style="
        display: flex; 
        align-items: center; 
        gap: 14px; 
        background: rgba(241, 245, 249, 0.85); 
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 14px 20px; 
        border-radius: 10px; 
        border: 1.5px solid #1e293b;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
        margin: 12px 0;
        transform: translateZ(0); /* Flattens 3D stacking context caused by backdrop-filter */
    ">
        <div class="loader-ring"></div>
        <span style="color: #0f172a; font-weight: 600; font-size: 15px; letter-spacing: 0.2px;">{message}</span>
    </div>
    <style>
        .loader-ring {{
            width: 22px;
            height: 22px;
            border: 3px solid #cbd5e1;
            border-top: 3px solid #2563eb;
            border-radius: 50%;
            box-sizing: border-box;
            transform-style: flat; /* Forces 2D plane rendering */
            will-change: transform;
            animation: custom-spin 0.8s linear infinite;
        }}
        @keyframes custom-spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    """,
        unsafe_allow_html=True,
    )


#####################################
#   Model Loaders (Silent Caching)  #
#####################################

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

@st.cache_resource(show_spinner=False)
def get_hugging_face():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


#############################
# Delete Namespace Function #
#############################
def del_namespace():
    if not st.session_state.userid:
        raise ValueError("Error: Invalid namespace, cannot delete!")

    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=384,
                spec= ServerlessSpec(cloud="aws",region="us-east-1"),
                metric="dotproduct",
                )
        index = pc.Index(index_name)
            
    except Exception as e:
        print(f"Error in initialising Pinecone database: \n\n{e}")
        raise RuntimeError(f"Error in initialising Pinecone database: \n{e}")
    
    try:
        stats = index.describe_index_stats()
        existing_namespaces = stats.get("namespaces",{})
        if(st.session_state.userid in existing_namespaces):
            print(F"Deleting namespace: {st.session_state.userid}")
            index.delete(delete_all=True, namespace=st.session_state.userid)
        else:
            print(f"Namespace {st.session_state.userid} does not exist.")
    
    except Exception as e:
        print(f"Error in deleting namespace\n\n{e}")
        raise RuntimeError(f"Error in deleting namespace\n{e}")

#########################
# Reset Review Function #
#########################
def reset_review_state():
    st.session_state.review_button = False
    st.session_state.submitted_url = ""
    
    try:
        del_namespace()
    except Exception as e:
        print(f"Error in deleting namespace: {e}")

    st.session_state.userid = f"user_{uuid.uuid4().hex[:10]}_{int(time.time())}"


#########################################################################################
#init models as cache so that it does not load again on every refresh or review button  #
#########################################################################################
if "models_loaded" not in st.session_state:
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        with st.status("⚙️ Initializing AI Engine...", expanded=True) as status:
            with custom_loader("🔄 Loading Gemini-3.1-Flash-Lite..."):
                llm = get_llm()
            
            with custom_loader("🔄 Loading HuggingFace Transformer All-MiniLM-L6-v2..."):
                dense_model = get_hugging_face()
            
            status.update(label="✅ Models loaded successfully!", state="complete", expanded=True)
            
    st.session_state.models_loaded = True
    time.sleep(0.5)  # Brief delay to let user see completion status
    st.rerun()       # Rerun to cleanly clear status UI from screen
else:
    # On reruns, retrieve instances instantly from Streamlit cache
    llm = get_llm()
    dense_model = get_hugging_face()



####################
#  Sidebar Panel   #
####################

with st.sidebar:
    st.markdown("### USER PROFILE")
    st.markdown(
        f"""
        <div class="user-card">
            <span style="font-size:0.8rem; color:#475569;">USER ID:</span><br>
            <strong style="color:#0F172A;">{st.session_state.userid}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Clear Tasks", use_container_width=True):
        with st.status("Clearing Session Data", expanded=True) as status:
            st.write("🧹🌲 Clearing data in pinecone vector store...")
            time.sleep(0.5)
            
            st.write("👤 Deleting namespace...")
            time.sleep(0.5)
            
            st.write("🆔✨Creating a new session User ID...")
            time.sleep(0.5)
            
        # Run reset state after spinners finish
        reset_review_state()
        st.rerun()


####################
#  Main Dashboard  # 
####################
left_pad, main_content_col, right_pad = st.columns([1, 3, 1])

with main_content_col:
    dashboard_container = st.empty()

    if not st.session_state.review_button:
        # Input Form View
        with dashboard_container.container():
            st.markdown(
                '<div class="main-title">AI Code Review Dashboard</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="sub-title">Enter your GitHub Pull Request URL to run agentic analysis</div>',
                unsafe_allow_html=True,
            )

            pr_url = st.text_input(
                "PR URL",
                placeholder="https://github.com/owner/repository/pull/12",
                label_visibility="collapsed",
            )

            st.markdown(
                "<div style='margin-top: 16px;'></div>",
                unsafe_allow_html=True,
            )

            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

            with btn_col2:
                if st.button("Review", use_container_width=True):
                    if pr_url:
                        st.session_state.submitted_url = pr_url
                        st.session_state.review_button = True
                        st.rerun()
                    else:
                        st.warning("Please enter a valid PR URL.")
    else:
        # Agent Output View
        with dashboard_container.container():
            st.markdown(
                '<div class="main-title">Agent Analysis Running</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="sub-title">Target: <code>{st.session_state.submitted_url}</code></div>',
                unsafe_allow_html=True,
            )

            ###############
            #  Main Logic #
            ###############
            
            #OUTPUT VIEW
            
            #loading animation
            spin = custom_loader("⚡ Dissecting your PR diff like a senior staff engineer 🔮")

            #report
            final_report = Optional[Any]

            #queries list
            queries_list = []

            #expanders
            expander_ingest = st.expander("Processing your pull request & generating review", expanded=True)
            expander_output = st.expander("Output", expanded=True)
            expander_queries = st.expander("Queries by LLM",expanded=False)

            #placeholder containers
            with expander_ingest:
                container = st.empty()
                container.spinner("⌛ Waiting to pull PR diff and repository chunks...📄")

            with expander_output:
                output_container = st.empty()
                output_loading = custom_loader("📊⏱️ Waiting for final report generation...🤖💭")

            with expander_queries:
                query_container = st.empty()

            try:
                app = code_review_ai_app(dense_model, llm, st.session_state.userid)

                #invoke the graph as a stream to print different stages
                for output in app.stream({"pr_url":st.session_state.submitted_url}):
                    for node_name, node_state in output.items(): #will recieve node names of workflow as executes

                        if node_name == "pull_diff":
                            container.empty()
                            with container.spinner("Fetching and Processing PR Diff 🌀"):
                                container.markdown(f"🔍 PR Diff Fetched\n\n🌿📄 Found {len(node_state.get('files_changed',[]))} changed files📥")

                        elif node_name == "ingestion_and_chunk":

                            container.empty()
                            container.success(
                            "✅ Repository code aware splitting, embedding, and Pinecone"
                            " upsert complete!"
                            )


                        elif node_name == "retrieve_and_crag":

                            query = node_state.get("current_query", "")
                            docs = node_state.get("retrieved_context", [])
                            count = node_state.get("counter", 1)

                            #append generated queries in list
                            queries_list.append(
                                f"**Query Attempt #{count}** 🔄\n\n"
                                f"**Query:** `{query}` 📝\n\n"
                            )

                            #display all queries in the queries container
                            query_container.empty()
                            with query_container.container():
                                for q_text in queries_list:
                                    st.markdown(q_text)


                        elif node_name == "grader_agent":

                            container.empty()
                            
                            is_rel = node_state.get('is_relevant',False)
                            if is_rel:
                                container.success("🌟Relevant check passed, documents are matching PR scope.")
                            else:
                                container.warning("⚠️ Relevance check failed! Rewriting the query.")


                        elif node_name == "review_agent":
                            container.empty()
                            final_report = node_state.get("final_report","")
                            if final_report:
                                container.success("🎉 Code review completed successfully! 🚀✨")
                            output_loading.empty()
                            spin.empty()

                            #stream the output
                            def stream_data():
                                for word in final_report.split(" "):
                                    yield word + " "
                                    time.sleep(0.01)

                            with expander_output:
                                output_container.write_stream(stream_data)                  


            except Exception as e:
                output_loading.empty() #error agaya to kya hi load karega bhai
                spin.empty() #error agaya to kya hi load karega bhai
                st.error(f"Error while executing the pipeline!\n\nException Details:- {e}\n\nYou can restart by clicking 'New Review' below")

            
            st.markdown(
                "<div style='margin-top: 16px;'></div>",
                unsafe_allow_html=True,
            )

            #new review button, will start a new session with new userid, will remove all data in pinecone
            nbtn_col1, nbtn_col2, nbtn_col3 = st.columns([1, 1, 1])
            with nbtn_col2:
                if st.button("New Review", use_container_width=True, on_click=reset_review_state):
                    pass