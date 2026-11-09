from google import genai

client = genai.Client(api_key="API_KEY")
MODEL = "gemini-3.5-flash-lite"

text_data = """
У програмуванні для збереження колекцій даних використовують різні структури. 
Масив (Array) має фіксований розмір та зберігає елементи одного типу даних, що забезпечує високу швидкість обробки в пам'яті. 
Список (List) є динамічною структурою даних, яка дозволяє змінювати розмір, додавати чи видаляти елементи та зберігати дані різних типів. 
Кортеж (Tuple) — це незмінна (immutable) послідовність, розмір та вміст якої не можна модифікувати після створення, що забезпечує захист даних від випадкових змін та високу ефективність.
"""

prompt_single = f"""
На основі наведеного тексту про структури даних згенеруй 2 тестові запитання з 4 варіантами відповідей (A, B, C, D) та вкажи правильну відповідь для кожного:

{text_data}
"""

interaction_single = client.interactions.create(
    model=MODEL,
    input=prompt_single
)

print("ОДИНОЧНИЙ PROMPT")
print(interaction_single.output_text)

prompt_step1 = f"""
Проаналізуй текст та виділи по 1-2 ключові відмінні характеристики для кожної структури даних (Масив, Список, Кортеж):

{text_data}
"""
interaction_step1 = client.interactions.create(
    model=MODEL,
    input=prompt_step1
)

print("\nДЕКОМПОЗИЦІЯ. КРОК 1: Виділення характеристик")
print(interaction_step1.output_text)

prompt_step2 = f"""
На основі цих виділених характеристик сформулюй 2 чіткі тестові запитання для перевірки знань різниці між масивом, списком та кортежем:

{interaction_step1.output_text}
"""
interaction_step2 = client.interactions.create(
    model=MODEL,
    input=prompt_step2
)

print("\nДЕКОМПОЗИЦІЯ. КРОК 2: Генерація запитань")
print(interaction_step2.output_text)

prompt_step3 = f"""
Для кожного із цих запитань розроби по 4 варіанти відповідей (A, B, C, D), де лише один є правильним, та наприкінці надай правильні відповіді з поясненням:

{interaction_step2.output_text}
"""
interaction_step3 = client.interactions.create(
    model=MODEL,
    input=prompt_step3
)

print("\nДЕКОМПОЗИЦІЯ. КРОК 3: Варіанти відповідей та ключі")
print(interaction_step3.output_text)