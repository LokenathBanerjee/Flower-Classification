import os

from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

DEFAULT_CHAT_MODEL      = "gemini-2.5-flash"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"

PROMPT_TEMPLATE = """You are a knowledgeable assistant answering questions about a specific flower.

Flower: {flower_name}

Use ONLY the context below to answer the user's question. If the context does not
contain enough information to answer, say plainly that you don't have that
information in the knowledge base for this flower — do not guess or use outside
knowledge.

Context:
{context}

Question: {question}

Answer in 2-4 sentences, in a friendly and informative tone."""


class FlowerRAG:
    """Wraps Chroma retrieval + a Gemini chat model into one .answer() call."""

    def __init__(
        self,
        persist_directory: str,
        api_key: str,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        if not api_key:
            raise ValueError(
                "No Gemini API key was found. Set GOOGLE_API_KEY in your .env file "
                "locally, or in the Streamlit Cloud secrets manager when deployed."
            )

        if not os.path.isdir(persist_directory) or not os.listdir(persist_directory):
            raise FileNotFoundError(
                f"No vector database found at '{persist_directory}'. "
                f"Run ingest.py once to build it from the PDFs in data/knowledge_base/ "
                f"before starting the app."
            )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model, google_api_key=api_key
        )
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        self.llm = ChatGoogleGenerativeAI(
            model=chat_model, google_api_key=api_key, temperature=0.3
        )

    def answer(self, flower_name: str, question: str, k: int = 4) -> dict:
        """
        Returns:
        {
            "answer": "...",
            "sources": ["PETUNIA.pdf", ...],
            "found_relevant_info": True/False
        }
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("Question is empty — nothing to search for.")

        augmented_query = f"{flower_name}: {question}"

        try:
            docs = self.vectorstore.similarity_search(augmented_query, k=k)
        except Exception as e:
            raise RuntimeError(f"Vector search failed: {e}")

        if not docs:
            return {
                "answer": (
                    f"The knowledge base returned no results for that query about {flower_name}. "
                    f"Make sure ingest.py has been run and the vector_db/ folder is not empty."
                ),
                "sources": [],
                "found_relevant_info": False,
            }

        context = "\n\n".join(doc.page_content for doc in docs)
        sources = sorted({
            os.path.basename(doc.metadata.get("source", "unknown"))
            for doc in docs
        })

        prompt = PROMPT_TEMPLATE.format(
            flower_name=flower_name, context=context, question=question
        )

        try:
            response = self.llm.invoke(prompt)
        except Exception as e:
            raise RuntimeError(
                f"Gemini API call failed: {e}. "
                f"Check your API key, quota, and internet connection."
            )

        return {
            "answer": response.content,
            "sources": sources,
            "found_relevant_info": True,
        }