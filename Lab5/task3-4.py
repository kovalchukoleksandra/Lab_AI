import os
import chromadb
from google import genai

client = genai.Client(api_key="API_KEY")
MODEL = "gemini-embedding-001"

script_dir = os.path.dirname(os.path.abspath(__file__))
chunks_dir = os.path.join(script_dir, "chunks_output")
chroma_db_dir = os.path.join(script_dir, "chroma_db")

def get_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model=MODEL,
        contents=text,
    )
    return response.embeddings[0].values

print("Завдання 3: Генерація Embeddings")

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

                print(f"Генерація вектора для {chunk_id} ({source})...")

                try:
                    vector = get_embedding(content)
                    
                    ids.append(chunk_id)
                    documents.append(content)
                    metadatas.append({"source": source})
                    embeddings.append(vector)
                except Exception as e:
                    print(f"Помилка створення вектора для {chunk_id}: {e}")

print("\nЗавдання 4: Збереження у ChromaDB")

chroma_client = chromadb.PersistentClient(path=chroma_db_dir)

collection = chroma_client.get_or_create_collection(name="python_docs")

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