import sys
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import CharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma 
from langchain_ollama import OllamaLLM 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Step 1. Define Constants 
# Defining the file path for the data and the embedding model
DATA_PATH = "speech.txt"
# Using a lightweight sentence transformer model for embeddings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 
# Using Mistral 7B model via Ollama for local LLM inference
LLM_MODEL = "mistral" 
VECTOR_DB_PATH = "chroma_db" # Directory to save the local vector store


def _format_docs(docs):
    """Join retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    """
    Main function to build and run the RAG Q&A system.
    """
    try:
        # Step 2. Load and Process Documents 
        print("Loading data...")
        # Load the document from the speech.txt file 
        loader = TextLoader(DATA_PATH, encoding = "utf-8")
        documents = loader.load()

        # Split the text into chunks 
        text_splitter = CharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
        texts = text_splitter.split_documents(documents)
        print(f"Loaded and split {len(texts)} text chunks.")

        # Step 3. Create Embeddings and Vector Store 
        print("Creating embeddings and vector store...")
        # Using the specified HuggingFace model for embeddings (runs locally) 
        embeddings = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL)

        # Create a ChromaDB vector store from the text chunks 
        # It will be persisted locally in the VECTOR_DB_PATH directory
        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory = VECTOR_DB_PATH
        )
        print(f"Vector store created at {VECTOR_DB_PATH}.")

        # Step 4. Setup LLM and Retriever 
        print("Setting up LLM and Retriever chain...")
        # Initialize the Ollama LLM (Mistral 7B) 
        # Setting a low temperature for more deterministic responses
        # temperature controls randomness in output
        llm = OllamaLLM(model = LLM_MODEL, temperature = 0.2)

        retriever = db.as_retriever()

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer using only the provided context."),
            ("human", "Question: {question}\n\nContext:\n{context}")
        ])

        qa_chain = (
            {"context": retriever | RunnableLambda(_format_docs), "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("Q&A system is ready. Ask a question (or type 'exit' to quit).")

        # Step 5. Run the Q&A Loop 
        while True:
            query = input("\n[Question]: ")
            if query.lower() == "exit":
                print("Exiting...")
                break
            
            try:
                # Retrieve relevant chunks and generate an answer 
                result = qa_chain.invoke(query)

                print("\n[Answer]:")
                print(result)

                # Optional: Print the source documents that were retrieved
                # sources = retriever.invoke(query)
                # print("\n[Sources]:")
                # for source in sources:
                #     print(f"- {source.page_content[:100]}...")
                    
            except Exception as e:
                print(f"Error during query processing: {e}")

    # Adding error handling for file not found and other exceptions
    except FileNotFoundError:
        print(f"Error: The file '{DATA_PATH}' was not found.")
        print("Please make sure 'speech.txt' is in the same directory as 'main.py'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please ensure Ollama is running and 'mistral' model is pulled.")
        print("Run 'ollama pull mistral' in your terminal.")
        sys.exit(1)

if __name__ == "__main__":
    main()