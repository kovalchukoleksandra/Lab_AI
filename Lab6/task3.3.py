import os
import chromadb
from google import genai

client = genai.Client(api_key="API_KEY")  

MODEL_EMBEDDING = "gemini-embedding-001"
MODEL_LLM = "gemini-3.5-flash-lite"

script_dir = os.path.dirname(os.path.abspath(__file__))
chroma_db_dir = os.path.join(script_dir, "chroma_db_lab6")

chroma_client = chromadb.PersistentClient(path=chroma_db_dir)
collection = chroma_client.get_collection(name="it_categories")


def get_query_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model=MODEL_EMBEDDING,
        contents=text
    )
    return response.embeddings[0].values


def classify_with_rag(text: str, top_k: int = 1):
    query_vector = get_query_embedding(text)
    
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    
    retrieved_doc = results["documents"][0][0]
    retrieved_meta = results["metadatas"][0][0]
    source_name = retrieved_meta["source"]
    source_category = retrieved_meta["category"]

    prompt = f"""Визнач категорію наведеного тексту, використовуючи контекст із бази знань.
Можливі категорії: програмування, штучний інтелект, бази даних, комп'ютерні мережі.
Поверни лише назву категорії.

Контекст із бази знань:
{retrieved_doc}

Текст для класифікації:
{text}"""

    try:
        response = client.models.generate_content(
            model=MODEL_LLM,
            contents=prompt
        )
        print(f"Знайдений файл: {source_name} (Категорія: {source_category})")
        print(f"Знайдений текст (чанк): {retrieved_doc}")
        print(f"Відповідь LLM (RAG): {response.text.strip()}")
    except Exception as e:
        print(f"Помилка RAG: {e}")


test_texts = [
    "Використання інкапсуляції та поліморфізму дозволяє створювати гнучку архітектуру класів.",
    "Механізм self-attention в архітектурі трансформерів допомагає моделі враховувати контекст слів.",
    "Зовнішні ключі забезпечують цілісність зв'язків між таблицями у реляційних СУБД.",
    "Маршрутизатори визначають оптимальний шлях передачі пакетів між підмережами за IP-адресами.",
    "Згорткові нейронні мережі ефективно виділяють візуальні ознаки для розпізнавання об'єктів на фото.",
    "Принципи ACID гарантують атомарність та надійність виконання транзакцій при збоях."
]

print("ЗАВДАННЯ 3. КЛАСИФІКАЦІЯ ТЕКСТІВ ЗА ДОПОМОГОЮ RAG ")
for idx, text in enumerate(test_texts, 1):
    print("\n")
    print(f"Приклад {idx}: {text}")
    classify_with_rag(text, top_k=1)