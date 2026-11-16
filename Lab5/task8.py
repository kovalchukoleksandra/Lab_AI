import os
import chromadb
from google import genai

client = genai.Client(api_key="API_KEY")  

MODEL_EMBEDDING = "gemini-embedding-001"
MODEL_LLM = "gemini-3.5-flash-lite"

script_dir = os.path.dirname(os.path.abspath(__file__))
chroma_db_dir = os.path.join(script_dir, "chroma_db")

chroma_client = chromadb.PersistentClient(path=chroma_db_dir)
collection = chroma_client.get_collection(name="python_docs")


def get_query_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model=MODEL_EMBEDDING,
        contents=text
    )
    return response.embeddings[0].values


def ask_without_rag(question: str):
    try:
        response = client.models.generate_content(
            model=MODEL_LLM,
            contents=question
        )
        print("\n 1. ВІДПОВІДЬ LLM БЕЗ RAG (зі своїх загальних знань):")
        print(response.text.strip())
    except Exception as e:
        print(f"Помилка LLM без RAG: {e}")


def ask_with_rag(question: str, top_k: int = 3):
    query_vector = get_query_embedding(question)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    
    retrieved_docs = results["documents"][0]
    retrieved_metas = results["metadatas"][0]
    
    context_parts = []
    sources_list = []
    
    for i in range(len(retrieved_docs)):
        doc_text = retrieved_docs[i]
        source_name = retrieved_metas[i]["source"]
        context_parts.append(f"[{source_name}]: {doc_text}")
        if source_name not in sources_list:
            sources_list.append(source_name)
            
    retrieved_context = "\n\n".join(context_parts)

    prompt = f"""Використовуючи наведений контекст, дай точну та зрозумілу відповідь на запитання користувача.

Контекст:
{retrieved_context}

Запитання:
{question}

Якщо відповіді на запитання немає в контексті, прямо повідом про це та не вигадуй інформацію."""

    try:
        response = client.models.generate_content(
            model=MODEL_LLM,
            contents=prompt
        )
        print("\n 2. ВІДПОВІДЬ LLM ІЗ ВИКОРИСТАННЯМ RAG (строго за документами):")
        print(response.text.strip())
        print(f"\nВикористані джерела: {', '.join(sources_list)}")
    except Exception as e:
        print(f"Помилка RAG: {e}")


compare_queries = [
    "Чим відрізняються списки (list) від кортежів (tuple) та як зберігають дані словники (dict)?",
    "Як працюють декоратори (@decorator) та асинхронні функції (async/await) у Python?",
    "Як налаштувати підключення до бази даних PostgreSQL за допомогою бібліотеки SQLAlchemy?"
]

print("ДОДАТКОВЕ ЗАВДАННЯ*. ПОРІВНЯННЯ LLM БЕЗ RAG ТА З RAG")
for q in compare_queries:
    print("\n")
    print(f"ЗАПИТАННЯ: {q}")
    ask_without_rag(q)
    print("\n")
    ask_with_rag(q, top_k=3)