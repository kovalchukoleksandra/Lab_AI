from google import genai
client = genai.Client(api_key="API_KEY")
MODEL = "gemini-3.5-flash-lite"

prompt_teacher = """
Ти — викладач з предмету ШІ.
Поясни, як працює штучний інтелект і на чому ґрунтується
його здатність навчатися.
"""
interaction_teacher = client.interactions.create(
    model=MODEL,
    input=prompt_teacher
)

print("РОЛЬ ВИКЛАДАЧА")
print(interaction_teacher.output_text)

prompt_programmer = """
Ти — програміст, який працює зі штучним інтелектом.
Поясни, як працює штучний інтелект і на чому ґрунтується
його здатність навчатися.
"""
interaction_programmer = client.interactions.create(
    model=MODEL,
    input=prompt_programmer
)

print("\nРОЛЬ ПРОГРАМІСТА")
print(interaction_programmer.output_text)


prompt_scientist = """
Ти — науковець, який досліджує штучний інтелект.
Поясни, як працює штучний інтелект і на чому ґрунтується
його здатність навчатися.
"""
interaction_scientist = client.interactions.create(
    model=MODEL,
    input=prompt_scientist
)

print("\nРОЛЬ НАУКОВЦЯ")
print(interaction_scientist.output_text)
