from google import genai
from google.genai import types
from task1 import my_tools
from task3 import PLANNING_INSTRUCTION

client = genai.Client(api_key="API_KEY")
MODEL_LLM = "gemini-3.5-flash-lite"

config = types.GenerateContentConfig(
    system_instruction=PLANNING_INSTRUCTION,
    tools=my_tools,
    temperature=0.3
)

agent_chat = client.chats.create(model=MODEL_LLM, config=config)

messages = [
    "Я вивчаю Python.",
    "Порадь мені 2 завдання для практики.",
    "Поясни, як виконати перше завдання, і порахуй через калькулятор, скільки хвилин піде на обидва, якщо на кожне треба по 35 хвилин."
]

print("ЗАВДАННЯ 5. ПЕРЕВІРКА ПАМ'ЯТІ (MEMORY)")
for i, msg in enumerate(messages, 1):
    print(f"\nПовідомлення {i}")
    print(f"Користувач: {msg}")
    response = agent_chat.send_message(msg)
    print(f"Агент:\n{response.text}")