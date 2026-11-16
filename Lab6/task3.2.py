import os
import chromadb
from google import genai

client = genai.Client(api_key="API_KEY")  
MODEL = "gemini-embedding-001"

script_dir = os.path.dirname(os.path.abspath(__file__))

chunks_dir = os.path.join(script_dir, "chunks_output_lab6")
chroma_db_dir = os.path.join(script_dir, "chroma_db_lab6")

FILE_TO_CATEGORY = {
    "programming.txt": "програмування",
    "artificial_intelligence.txt": "штучний інтелект",
    "databases.txt": "бази даних",
    "computer_networks.txt": "комп'ютерні мережі"
}


def get_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model=MODEL,
        contents=text,
    )
    return response.embeddings[0].values


print("Генерація Embeddings для Лабораторної роботи № 6")

ids = []
documents = []
metadatas = []
embeddings = []

if os.path.exists(chunks_dir):
    chunk_files = sorted(os.listdir(chunks_dir))

    for file_name in chunk_files:
        if file_name.endswith(".txt"):
            file_path = os.path.join(chunks_dir, file_name)

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

                source = lines[0].replace("Source:", "").strip()
                chunk_id = lines[1].replace("ID:", "").strip()
                content = "".join(lines[4:]).strip()
                category = FILE_TO_CATEGORY.get(source, "невідомо")

                print(f"Генерація вектора для {chunk_id} ({source} -> {category})")

                try:
                    vector = get_embedding(content)
                    
                    ids.append(chunk_id)
                    documents.append(content)
                    metadatas.append({
                        "source": source,
                        "category": category
                    })
                    embeddings.append(vector)
                except Exception as e:
                    print(f"Помилка створення вектора для {chunk_id}: {e}")
else:
    print(f"Помилка: папку {chunks_dir} не знайдено!")

print("\nЗбереження у нову базу ChromaDB")

chroma_client = chromadb.PersistentClient(path=chroma_db_dir)

collection = chroma_client.get_or_create_collection(name="it_categories")

if ids:
    try:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"\nУспішно збережено {collection.count()} об'єктів у векторну базу ChromaDB!")
    except Exception as e:
        print(f"Помилка збереження у ChromaDB: {e}")