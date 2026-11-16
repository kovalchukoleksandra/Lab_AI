import time
from google import genai

client = genai.Client(api_key="API_KEY")  
MODEL_LLM = "gemini-3.5-flash-lite"

def classify_prompt_engineering(text: str):
    prompt = f"""Визнач категорію наведеного тексту.
Можливі категорії: програмування, штучний інтелект, бази даних, комп'ютерні мережі.
Поверни лише назву категорії.

Текст: {text}"""

    try:
        response = client.models.generate_content(
            model=MODEL_LLM,
            contents=prompt
        )
        print(f"Відповідь LLM (Prompt Engineering): {response.text.strip()}")
    except Exception as e:
        print(f"Помилка: {e}")


test_texts = [
    "Використання інкапсуляції та поліморфізму дозволяє створювати гнучку архітектуру класів.",
    "Механізм self-attention в архітектурі трансформерів допомагає моделі враховувати контекст слів.",
    "Зовнішні ключі забезпечують цілісність зв'язків між таблицями у реляційних СУБД.",
    "Маршрутизатори визначають оптимальний шлях передачі пакетів між підмережами за IP-адресами.",
    "Згорткові нейронні мережі ефективно виділяють візуальні ознаки для розпізнавання об'єктів на фото.",
    "Принципи ACID гарантують атомарність та надійність виконання транзакцій при збоях."
]

print("ЗАВДАННЯ 2. КЛАСИФІКАЦІЯ ЗА ДОПОМОГОЮ PROMPT ENGINEERING")
for idx, text in enumerate(test_texts, 1):
    print("\n")
    print(f"Приклад {idx}: {text}")
    classify_prompt_engineering(text)
    time.sleep(1)