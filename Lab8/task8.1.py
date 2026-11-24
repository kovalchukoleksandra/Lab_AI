from collections import Counter
from pathlib import Path
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("TextAnalyzerMCP")
BASE_DIR = Path(__file__).parent

@mcp.tool()
def read_and_count_file(filename: str) -> str:
    """Читає текстовий файл з диска та повертає його текст і точну кількість слів та символів."""
    file_path = BASE_DIR / filename
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        words = len(content.split())
        chars = len(content)
        return (
            f"Вміст файлу '{filename}':\n{content}\n\n"
            f"[Статистика інструмента]: Слів — {words}, символів — {chars}."
        )
    except FileNotFoundError:
        return f"Помилка: файл '{filename}' не знайдено."

@mcp.tool()
def most_frequent_word(filename: str) -> str:
    """Знаходить найчастіше вживане слово (довжиною від 4 символів) у текстовому файлі."""
    file_path = BASE_DIR / filename
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().lower()
        words = [w.strip(".,!?:;()\"'—") for w in content.split()]
        filtered_words = [w for w in words if len(w) >= 4]
        if not filtered_words:
            return "У файлі недостатньо слів для аналізу."
        word, count = Counter(filtered_words).most_common(1)[0]
        return f"Найчастіше слово у файлі '{filename}': '{word}' (зустрічається {count} разів)."
    except FileNotFoundError:
        return f"Помилка: файл '{filename}' не знайдено."

if __name__ == "__main__":
    mcp.run()