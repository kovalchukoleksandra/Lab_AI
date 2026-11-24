import asyncio
import sys
from pathlib import Path
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

client = genai.Client(api_key="API_KEY")
MODEL_LLM = "gemini-3.5-flash-lite"

BASE_DIR = Path(__file__).parent

server_params = StdioServerParameters(
    command=sys.executable,
    args=[str(BASE_DIR / "task8.1.py")],
)

async def run_mcp_demo():
    print("ЗАВДАННЯ 8. ДЕМОНСТРАЦІЯ РОБОТИ MCP\n")

    user_query = (
        "Прочитай файл database.txt за допомогою MCP-інструмента read_and_count_file "
        "та визнач найчастіше слово за допомогою інструмента most_frequent_word. "
        "Виконай повний аналіз отриманого тексту:\n"
        "1) вкажи точну кількість слів, символів та найчастіше слово, використовуючи статистику інструментів;\n"
        "2) визнач тему тексту;\n"
        "3) виділи ключові ідеї;\n"
        "4) сформуй короткий висновок;\n"
        "5) перевір коректність свого результату за оригінальним текстом."
    )

    print(f"1. Запит користувача:\n{user_query}\n")

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                mcp_tools = await session.list_tools()
                print(
                    "2. MCP Client виявив доступні інструменти:",
                    [tool.name for tool in mcp_tools.tools],
                    "\n"
                )

                async def read_and_count_file(filename: str) -> str:
                    """Читає текстовий файл з диска через MCP-сервер та повертає його вміст і статистику слів та символів."""
                    print(f"3.1. AI-модель -> MCP: виклик інструмента 'read_and_count_file' для файлу '{filename}'")
                    result = await session.call_tool("read_and_count_file", arguments={"filename": filename})
                    tool_output = result.content[0].text
                    print(f"4.1. MCP -> Результат від 'read_and_count_file':\n{tool_output}\n")
                    return tool_output

                async def most_frequent_word(filename: str) -> str:
                    """Знаходить найчастіше вживане слово у текстовому файлі через MCP-сервер."""
                    print(f"3.2. AI-модель -> MCP: виклик інструмента 'most_frequent_word' для файлу '{filename}'")
                    result = await session.call_tool("most_frequent_word", arguments={"filename": filename})
                    tool_output = result.content[0].text
                    print(f"4.2. MCP -> Результат від 'most_frequent_word':\n{tool_output}\n")
                    return tool_output

                response = await client.aio.models.generate_content(
                    model=MODEL_LLM,
                    contents=user_query,
                    config=types.GenerateContentConfig(
                        tools=[read_and_count_file, most_frequent_word],
                        temperature=0.2
                    )
                )

                print(
                    "5. Фінальна відповідь AI-моделі "
                    "(з урахуванням даних від MCP):\n"
                    f"{response.text}"
                )

    finally:
        await client.aio.aclose()

if __name__ == "__main__":
    asyncio.run(run_mcp_demo())