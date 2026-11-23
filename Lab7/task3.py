from google import genai
from google.genai import types
from task1 import my_tools
import time

client = genai.Client(api_key="API_KEY")
MODEL_LLM = "gemini-3.5-flash-lite"  

PLANNING_INSTRUCTION = """
Ти — AI-помічник студента.
Для складних або обчислювальних завдань використовуй доступні інструменти (calculator для обчислень, days_until_exam для дат у форматі YYYY-MM-DD).

Після отримання всіх необхідних даних від інструментів сформуй цілісну відповідь суворо за такою структурою (ніколи не пропускай перший блок):

ПЛАН ДІЙ:
1. ...
2. ...
3. ...

ВИКОНАННЯ ПЛАНУ:
1. ...
2. ...
3. ...

РЕЗУЛЬТАТ:
...

Важливо:
- План повинен відповідати конкретному запиту користувача, а не бути шаблонним.
- У розділі «РЕЗУЛЬТАТ» використовуй точні дані, отримані від інструментів.
- Якщо запит простий теоретичний (на 1-2 речення без обчислень), розділ «ПЛАН ДІЙ» можна не виводити.
"""

def ask_planning_agent(user_query: str):
    config = types.GenerateContentConfig(
        system_instruction=PLANNING_INSTRUCTION,
        tools=my_tools,
        temperature=0.2
    )
    
    try:
        chat = client.chats.create(model=MODEL_LLM, config=config)
        response = chat.send_message(user_query)
        print(f"Запит: {user_query}\n")
        print(f"Відповідь агента:\n{response.text}\n")
        time.sleep(2)
    except Exception as e:
        print(f"Помилка агента: {e}")

if __name__ == "__main__":
    print("ЗАВДАННЯ 4. РЕАЛІЗАЦІЯ ПЛАНУВАННЯ\n")
    ask_planning_agent("Обчисли середнє арифметичне чисел 12, 18, 25, 30 та поясни результат.")
    ask_planning_agent("Мої оцінки за модулі з дисципліни 'Комп'ютерна графіка': 90, 95, 100. Порахуй мій середній бал, дізнайся скільки днів залишилось до іспиту 2026-12-14 і склади короткий план підготовки.")