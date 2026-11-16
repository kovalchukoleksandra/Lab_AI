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

def ask_rag(question: str, top_k: int = 3):
    print(f"\n")
    print(f"ЗАПИТАННЯ: {question}")
    
    query_vector = get_query_embedding(question)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    
    retrieved_docs = results["documents"][0]
    retrieved_metas = results["metadatas"][0]
    retrieved_ids = results["ids"][0]
    
    print("\nЗАВДАННЯ 5. РЕЗУЛЬТАТИ RETRIEVAL (ЗНАЙДЕНІ ФРАГМЕНТИ):")
    context_parts = []
    sources_list = []
    
    for i in range(len(retrieved_docs)):
        doc_text = retrieved_docs[i]
        source_name = retrieved_metas[i]["source"]
        chunk_id = retrieved_ids[i]
        
        print(f"  [{i+1}] {chunk_id} (Джерело: {source_name}) -> {doc_text[:100]}...")
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
        answer = response.text.strip()
        print("\nЗАВДАННЯ 6. ФІНАЛЬНА ВІДПОВІДЬ LLM:")
        print(answer)
        print(f"\nВикористані джерела: {', '.join(sources_list)}")
    except Exception as e:
        print(f"Помилка генерації відповіді LLM: {e}")

test_queries = [
    "Які базові типи даних існують у Python та який тип повертає функція input()?",
    "Чим відрізняються списки (list) від кортежів (tuple) та як зберігають дані словники (dict)?",
    "За допомогою якого ключового слова оголошуються функції в Python та для чого використовується інструкція return?",
    "Що таке конструктор класу __init__ та які три основні стовпи ООП виділяють у Python?",
    "Для чого використовується менеджер пакетів pip та як створити віртуальне середовище через venv?",
    "Як працюють декоратори (@decorator) та асинхронні функції (async/await) у Python?",
    "Як налаштувати підключення до бази даних PostgreSQL за допомогою бібліотеки SQLAlchemy?"
]

print("ЗАВДАННЯ 5 ТА 6. ЗАПУСК ПОВНОГО ЦИКЛУ RAG")
print(f"Завантажено об'єктів із бази: {collection.count()}")

for q in test_queries:
    ask_rag(q, top_k=3)