import time
from agents import run_agent
from task4_5 import analyze_text

test_text = (
    "Основи програмування на Python базуються на використанні змінних із динамічною типізацією, де тип даних "
    "визначається автоматично під час присвоєння значення. Для роботи з наборами даних у Python застосовуються "
    "вбудовані структури, такі як списки (list) та словники (dict), а повторювані дії реалізуються через цикли "
    "for і while. Розбиття коду на окремі функції за допомогою ключового слова def робить програму читабельною "
    "та дозволяє повторно використовувати логіку без дублювання."
)

SINGLE_AGENT_INSTRUCTION = """
Ти — універсальний аналітик тексту. Проаналізуй наданий текст:
1) визнач тему;
2) виділи ключові ідеї;
3) сформуй короткий висновок;
4) перевір результат.
"""

print("ЗАВДАННЯ 6. ПОРІВНЯННЯ ОДНОГО ТА ДЕКІЛЬКОХ АГЕНТІВ\n")

print("1. ОДИН AI-АГЕНТ")
start_single = time.time()
single_res = run_agent(SINGLE_AGENT_INSTRUCTION, f"Проаналізуй текст:\n\n{test_text}", temperature=0.2)
time_single = round(time.time() - start_single, 2)
print(f"Час виконання: {time_single} сек.")
print(f"Результат:\n{single_res}\n")

time.sleep(5)

print("2. МУЛЬТИАГЕНТНА СИСТЕМА ")
start_multi = time.time()
plan, draft, multi_res = analyze_text(test_text)
time_multi = round(time.time() - start_multi, 2)
print(f"Час виконання: {time_multi} сек.")
print(f"Фінальний результат:\n{multi_res}")