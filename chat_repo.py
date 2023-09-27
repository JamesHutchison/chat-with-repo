import functools
import pickle
import os
from pathlib import Path
from langchain.document_loaders import TextLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
import streamlit as st
from langchain.vectorstores import FAISS

import langchain

CACHE_DIR = Path(".repo_chat")
CACHE_FILE = CACHE_DIR / "docs.pickle"

DIR_LIST = ["add_dirs_here"]
FILE_EXTENSIONS = {"ts", "tsx", "py", "json", "js", "jsx", "html", "md", "css"}


langchain.debug = True

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")


@functools.lru_cache(1)
def get_deeplake_db():
    my_activeloop_org_id = "your-org-id"
    my_activeloop_dataset_name = "your-dataset-name"
    dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

    return DeepLake(dataset_path=dataset_path, embedding_function=embeddings)


@st.cache_resource()
def in_memory_vectorstore() -> FAISS:
    import faiss
    from langchain.docstore import InMemoryDocstore

    embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    embedding_size = 1536

    index = faiss.IndexFlatL2(embedding_size)
    return FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


def do_load(db) -> None:
    dir_list = DIR_LIST
    docs = []
    file_extensions = FILE_EXTENSIONS
    ignored_extensions = set()

    for root_dir in dir_list:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                file_path = os.path.join(dirpath, file)

                if (
                    file_extensions
                    and (extension := os.path.splitext(file)[1][1:])
                    not in file_extensions
                ):
                    if extension not in ignored_extensions:
                        ignored_extensions.add(extension)
                        print(f"Ignoring file extension {extension}")
                    continue

                splitter_extension = extension
                if extension in ("ts", "tsx", "jsx", "json"):
                    splitter_extension = "js"
                if extension in ("md", "css"):
                    splitter_extension = None

                # loader = RecursiveCharacterTextSplitter(file_path, encoding="utf-8")
                # docs.extend(loader.load_and_split())
                CHUNK_SIZE = 1000
                if splitter_extension:
                    text_splitter = RecursiveCharacterTextSplitter.from_language(
                        splitter_extension, chunk_size=CHUNK_SIZE
                    )
                else:
                    text_splitter = CharacterTextSplitter(
                        separator="\n\n", chunk_size=CHUNK_SIZE
                    )
                docs.extend(
                    text_splitter.create_documents(
                        [Path(file_path).read_text()], metadatas=[{"source": file_path}]
                    )
                )
                print(f"Num docs: {len(docs)}")

    # from langchain.text_splitter import CharacterTextSplitter

    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # splitted_text = text_splitter.split_documents(docs)

    print("Loading documents...")
    cache_docs(docs)
    db.add_documents(docs)


def cache_docs(docs):
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir()

    pickle.dump(docs, CACHE_FILE.open("wb"))


def do_load_of_cached_embeddings(db):
    assert CACHE_FILE.exists()

    docs = pickle.load(CACHE_FILE.open("rb"))

    db.add_documents(docs)


@st.cache_resource()
def do_load_in_memory(db):
    if Path(CACHE_DIR).exists():
        do_load_of_cached_embeddings(db)
    else:
        do_load(db)


def do_streamlit(in_memory: bool) -> None:
    from streamlit_chat import message

    if in_memory:
        print("Using in-memory vectorstore")
        db = in_memory_vectorstore()
        do_load_in_memory(db)
    else:
        print("using deeplake vectorstore")
        db = get_deeplake_db()
    retriever = db.as_retriever()

    # Set the search parameters for the retriever
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 100
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 10

    # Create a ChatOpenAI model instance
    # model = ChatOpenAI(model_name="gpt-4")
    model = ChatOpenAI()

    # Create a RetrievalQA instance from the model and retriever
    qa_chain = RetrievalQAWithSourcesChain.from_llm(model, retriever=retriever)

    # Return the result of the query
    # qa.run("What is the repository's name?")

    # pip install streamlit streamlit_chat

    # Set the title for the Streamlit app
    st.title(f"Chat with Code Repository")

    # Initialize the session state for placeholder messages.
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["ready"]

    if "past" not in st.session_state:
        st.session_state["past"] = ["hello"]

    chat_history_container = st.container()

    # A field input to receive user queries
    user_input = st.text_input("", key="input")
    send = st.button("Send")

    with chat_history_container:
        # Search the databse and add the responses to state
        if send and user_input:
            output = qa_chain(user_input, return_only_outputs=True)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

        # Create the conversational UI using the previous states
        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["generated"][i], key=str(i))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--load-deeplake", action="store_true")
    parser.add_argument("--in-memory", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        if CACHE_DIR.exists():
            for file in CACHE_DIR.iterdir():
                file.unlink()
        CACHE_DIR.rmdir()

    if args.load_deeplake:
        do_load(in_memory=False)
    else:
        do_streamlit(args.in_memory)
