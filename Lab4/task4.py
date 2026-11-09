import json
from google import genai

client = genai.Client(api_key="API_KEY")
MODEL = "gemini-3.5-flash-lite"

input_text = """
Пошук у ширину (BFS) використовує чергу (FIFO) та гарантує знаходження найкоротшого шляху у незваженому графі.
Пошук у глибину (DFS) реалізується за допомогою стека (LIFO) або рекурсії та використовує менше пам'яті на графах із великою шириною.
BFS є абсолютно кращим алгоритмом за DFS, і його слід використовувати завжди, оскільки рекурсія в DFS лише псує структуру коду.
Часова складність обох алгоритмів становить O(V + E), де V — кількість вершин, а E — кількість ребер.
"""

prompt_json = f"""
Проаналізуй наведений текст про алгоритми пошуку на графах (DFS та BFS) та сформуй результат СУВОРНО у форматі JSON.
Не додавай жодного вступного чи підсумкового тексту, а також тегів markdown (без ```json чи ```).

Структура JSON має бути такою:
{{
  "topic": "Тема аналізу",
  "facts": [
    "Об'єктивний факт 1",
    "Об'єктивний факт 2"
  ],
  "opinions": [
    "Суб'єктивна думка 1",
    "Суб'єктивна думка 2"
  ],
  "conclusion": "Короткий підсумковий висновок"
}}

Текст для аналізу:
{input_text}
"""

interaction_json = client.interactions.create(
    model=MODEL,
    input=prompt_json
)

raw_output = interaction_json.output_text.strip()

if raw_output.startswith("```"):
    raw_output = raw_output.split("\n", 1)[1].rsplit("\n", 1)[0].strip()

print("РЕЗУЛЬТАТ JSON ВІД МОДЕЛІ")
print(raw_output)

print("\nРОЗБІР (ПАРСИНГ) ЗАСОБАМИ PYTHON ")
try:
    parsed_json = json.loads(raw_output)
    print(f"Статус: Успішно розпарсено! Тип об'єкта: {type(parsed_json)}")
    print(f"Тема: {parsed_json.get('topic')}")
    
    print("\nЗнайдені об'єктивні факти:")
    for fact in parsed_json.get("facts", []):
        print(f" • {fact}")
        
    print("\nЗнайдені суб'єктивні думки:")
    for opinion in parsed_json.get("opinions", []):
        print(f" • {opinion}")
        
    print(f"\nВисновок: {parsed_json.get('conclusion')}")
except json.JSONDecodeError as e:
    print(f"Помилка парсингу JSON: {e}")