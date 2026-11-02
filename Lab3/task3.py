from google import genai
client = genai.Client(api_key="API_KEY")
MODEL = "gemini-3.5-flash-lite"

prompt_initial = """
Розкажи про вплив штучного інтелекту на майбутнє професій.
"""
interaction_initial = client.interactions.create(
    model=MODEL,
    input=prompt_initial
)

print("ПОЧАТКОВИЙ PROMPT")
print(interaction_initial.output_text)

prompt_improved = """
Ти — фахівець із сучасних технологій та ринку праці.
Поясни, як штучний інтелект впливає на майбутнє професій
та які зміни він може принести у сферу зайнятості.
Зверни увагу на такі аспекти:
Які професії можуть змінитися під впливом ШІ;
Які нові можливості відкриваються для працівників;
Які навички будуть важливими в майбутньому.

Пояснення має бути призначене для студентів.
Використовуй прості та зрозумілі слова,
наводь конкретні приклади.

Обсяг відповіді — приблизно 200 слів.
Оформи відповідь у вигляді короткого вступу
та нумерованого списку основних змін.
"""

interaction_improved = client.interactions.create(
    model=MODEL,
    input=prompt_improved
)

print("\nВДОСКОНАЛЕНИЙ PROMPT")
print(interaction_improved.output_text)
