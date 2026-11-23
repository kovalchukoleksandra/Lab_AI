from google import genai
from google.genai import types
from task1 import my_tools

client = genai.Client(api_key="API_KEY")
MODEL_LLM = "gemini-3.5-flash-lite"  

def ask_basic_agent(user_query: str):
    config = types.GenerateContentConfig(
        system_instruction="Ти — AI-помічник студента. Якщо запит вимагає обчислень або роботи з датами, обов'язково використовуй доступні інструменти.",
        tools=my_tools,
        temperature=0.2
    )
    
    try:
        chat = client.chats.create(model=MODEL_LLM, config=config)
        response = chat.send_message(user_query)
        print(f"Запит: {user_query}")
        print(f"Відповідь агента:\n{response.text}\n")
    except Exception as e:
        print(f"Помилка агента: {e}")

print("ЗАВДАННЯ 3. РОБОТА БАЗОВОГО АГЕНТА")
ask_basic_agent("Що таке поліморфізм в ООП простими словами?")
ask_basic_agent("Обчисли середнє арифметичне чисел 12, 18, 25, 30.")