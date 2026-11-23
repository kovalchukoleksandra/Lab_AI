from datetime import datetime
import math

def calculator(expression: str) -> str:
    """
    Виконує математичні обчислення. Використовуй цей інструмент для арифметики,
    підрахунку середнього балу або розв'язання числових виразів.
    
    Args:
        expression: Математичний вираз, наприклад '(12 + 18 + 25 + 30) / 4' або '25 * 4'.
    Returns:
        Рядок із числовим результатом або повідомленням про помилку.
    """
    try:
        allowed = {"math": math, "abs": abs, "round": round}
        result = eval(expression, {"__builtins__": None}, allowed)
        return str(result)
    except ZeroDivisionError:
        return "Помилка: ділення на нуль неможливе."
    except Exception as e:
        return f"Помилка обчислення: {e}"

def days_until_exam(exam_date: str) -> str:
    """
    Обчислює, скільки днів залишилося від сьогодні до дати іспиту чи здачі лабораторної.
    
    Args:
        exam_date: Дата у форматі 'YYYY-MM-DD' (наприклад, '2026-12-15').
    Returns:
        Текстовий результат із кількістю днів.
    """
    try:
        target = datetime.strptime(exam_date, "%Y-%m-%d").date()
        today = datetime.now().date()
        delta = (target - today).days
        if delta < 0:
            return f"Дата {exam_date} вже минула ({abs(delta)} дн. тому)."
        elif delta == 0:
            return "Дедлайн вже сьогодні!"
        return f"До {exam_date} залишилося {delta} дн."
    except ValueError:
        return "Неправильний формат дати. Потрібен формат YYYY-MM-DD."

my_tools = [calculator, days_until_exam]