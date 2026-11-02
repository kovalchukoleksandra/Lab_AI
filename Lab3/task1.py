from google import genai
client = genai.Client(api_key="API_KEY")
MODEL = "gemini-3.5-flash-lite"

prompt_1 = """
Поясни, що таке штучний інтелект.
Опиши основний принцип його роботи та наведи приклади використання.
"""

interaction_1 = client.interactions.create(
    model=MODEL,
    input=prompt_1
)

print("ЗАПИТ 1. ПОЯСНЕННЯ")
print(interaction_1.output_text)

prompt_2 = """
Напиши невеликий текст на тему
"Вплив штучного інтелекту на освітній процес та навчання учнів".
Розкрий, як штучний інтелект допомагає учням засвоювати
навчальний матеріал, виконувати завдання та отримувати
індивідуальні рекомендації для покращення знань.
Обсяг тексту — приблизно 100 слів.
"""

interaction_2 = client.interactions.create(
    model=MODEL,
    input=prompt_2
)

print("\nЗАПИТ 2. ГЕНЕРАЦІЯ ТЕКСТУ")
print(interaction_2.output_text)

prompt_3 = """
Переклади українською мовою такий текст:

"Artificial intelligence is transforming modern technology.
It helps people automate routine tasks, analyze large amounts
of data and make better decisions."
"""

interaction_3 = client.interactions.create(
    model=MODEL,
    input=prompt_3
)

print("\nЗАПИТ 3. ПЕРЕКЛАД")
print(interaction_3.output_text)


prompt_4 = """
Скороти та узагальни наведений текст,
залишивши тільки основні думки:

"Штучний інтелект активно використовується в різних сферах
діяльності людини. У медицині він допомагає аналізувати
медичні зображення та виявляти захворювання.
В освіті штучний інтелект використовується для створення
індивідуальних навчальних програм.
У промисловості він дозволяє автоматизувати виробничі
процеси та контролювати якість продукції.
Також технології штучного інтелекту застосовуються
у транспорті, фінансах та інформаційній безпеці."
"""

interaction_4 = client.interactions.create(
    model=MODEL,
    input=prompt_4
)

print("\nЗАПИТ 4. УЗАГАЛЬНЕННЯ")
print(interaction_4.output_text)

prompt_5 = """
Сформуй список із 5 переваг використання штучного інтелекту
в освіті. Для кожної переваги додай коротке пояснення.
"""

interaction_5 = client.interactions.create(
    model=MODEL,
    input=prompt_5
)

print("\nЗАПИТ 5. ГЕНЕРАЦІЯ СПИСКУ")
print(interaction_5.output_text)
