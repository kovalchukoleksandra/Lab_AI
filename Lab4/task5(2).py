import pypdf
from google import genai
import os
client = genai.Client(api_key="API_KEY")
MODEL = "gemini-3.5-flash-lite"

script_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(script_dir, "КГ.pdf")

reader = pypdf.PdfReader(pdf_path)

extracted_text = ""
for page in reader.pages[:20]: 
    extracted_text += page.extract_text() + "\n"

print(f"Усього зчитано слів із PDF: {len(extracted_text.split())}")

prompt_direct = f"Проаналізуй наданий текст та зроби його детальне та точно структуроване узагальнення:\n\n{extracted_text}"

try:
    interaction_direct = client.interactions.create(
        model=MODEL,
        input=prompt_direct
    )
    direct_result = interaction_direct.output_text.strip()
    print("\n5.1 ОБРОБКА БЕЗ CHUNKING (DIRECT PROCESSING)")
    print(direct_result)
except Exception as e:
    direct_result = f"Помилка Direct Processing (можливо, перевищено ліміт контексту): {e}"
    print(direct_result)

words = extracted_text.split()
chunk_size = 500  
chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

print(f"\n5.2 CHUNKING (MAP-REDUCE). Текст розділено на {len(chunks)} chunks")

summaries_chunking = []
for idx, chunk in enumerate(chunks, 1):
    prompt_map = f"Зроби коротке та точне узагальнення цього фрагмента книги (Chunk {idx}):\n\n{chunk}"
    interaction_map = client.interactions.create(model=MODEL, input=prompt_map)
    summary = interaction_map.output_text.strip()
    summaries_chunking.append(summary)
    print(f"\nSummary for Chunk {idx}:\n{summary[:150]}...") 

combined_chunking = "\n\n".join([f"Частина {i+1}: {s}" for i, s in enumerate(summaries_chunking)])
prompt_reduce_chunking = f"На основі наступних часткових узагальнень сформулюй один цілісний підсумковий висновок для всього документа:\n\n{combined_chunking}"

interaction_reduce_chunking = client.interactions.create(
    model=MODEL,
    input=prompt_reduce_chunking
)

chunking_result = interaction_reduce_chunking.output_text.strip()
print(f"\n>>> ФІНАЛЬНИЙ РЕЗУЛЬТАТ (MAP-REDUCE CHUNKING):\n{chunking_result}")

window_size = 500
overlap = 100
step = window_size - overlap
sliding_chunks = []

for i in range(0, len(words), step):
    chunk_words = words[i:i + window_size]
    sliding_chunks.append(" ".join(chunk_words))
    if i + window_size >= len(words):
        break

print(f"\nSLIDING WINDOW. Текст розділено на {len(sliding_chunks)} sliding chunks")

summaries_sliding = []
for idx, chunk in enumerate(sliding_chunks, 1):
    prompt_sliding = f"Зроби узагальнення цього фрагмента книги з урахуванням суміжного контексту:\n\n{chunk}"
    interaction_sliding = client.interactions.create(model=MODEL, input=prompt_sliding)
    summary = interaction_sliding.output_text.strip()
    summaries_sliding.append(summary)

combined_sliding = "\n\n".join([f"Фрагмент {i+1}: {s}" for i, s in enumerate(summaries_sliding)])
prompt_reduce_sliding = f"На основі наступних часткових узагальнень (зроблених із перекриттям контексту) сформулюй остаточне підсумкове узагальнення:\n\n{combined_sliding}"

interaction_reduce_sliding = client.interactions.create(
    model=MODEL,
    input=prompt_reduce_sliding
)

sliding_result = interaction_reduce_sliding.output_text.strip()
print(f"\n>>> ФІНАЛЬНИЙ РЕЗУЛЬТАТ (SLIDING WINDOW):\n{sliding_result}")

prompt_compare = f"""
Проаналізуй та порівняй три підходи до узагальнення навчального посібника:

1. Пряма обробка без chunking (Direct):
{direct_result}

2. Метод Chunking (Map-Reduce без перекриття):
{chunking_result}

3. Метод Sliding Window (з перекриттям контексту):
{sliding_result}

Оціни їх за наступними критеріями:
- Повнота охоплення матеріалу та точність
- Збереження важливих фактів і деталей
- Структурованість підсумкового тексту
- Виявлення пропущеної інформації

Сформулюй чіткий висновок щодо доцільності використання кожного методу під час роботи з великими документами.
"""

interaction_compare = client.interactions.create(
    model=MODEL,
    input=prompt_compare
)

print("\n5.3 ПОРІВНЯЛЬНИЙ АНАЛІЗ РЕЗУЛЬТАТІВ")
print(interaction_compare.output_text.strip())