import os

script_dir = os.path.dirname(os.path.abspath(__file__))

doc_files = [
    "programming.txt",
    "artificial_intelligence.txt",
    "databases.txt",
    "computer_networks.txt"
]

chunks = []
sources = []

for file_name in doc_files:
    file_path = os.path.join(script_dir, file_name)
    
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            
            for p in paragraphs:
                chunks.append(p)
                sources.append(file_name)
    else:
        print(f"Увага: файл {file_name} не знайдено!")

output_dir = os.path.join(script_dir, "chunks_output_lab6")
os.makedirs(output_dir, exist_ok=True)

for idx, (chunk, source) in enumerate(zip(chunks, sources), 1):
    chunk_id = f"chunk_{idx:02d}"
    out_file_name = f"{chunk_id}.txt"
    out_file_path = os.path.join(output_dir, out_file_name)
    
    with open(out_file_path, "w", encoding="utf-8") as f:
        f.write(f"Source: {source}\n")
        f.write(f"ID: {chunk_id}\n")
        f.write("\n\n")
        f.write(chunk)

print(f"Успішно створено та збережено {len(chunks)} chunks у папку:\n{output_dir}")