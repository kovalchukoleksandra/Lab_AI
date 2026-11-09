from google import genai

client = genai.Client(api_key="API_KEY")
MODEL = "gemini-3.5-flash-lite"

large_text = """
Структури даних є базовим фундаментом комп'ютерної науки та розробки програмного забезпечення. 
Серед лінійних структур даних особливе місце посідають списки (List), черги (Queue) та стеки (Stack), 
кожна з яких має унікальні характеристики організації пам'яті, швидкодії та алгоритмічного призначення. 
Список (List) є однією з найгнучкіших та найпоширеніших структур даних у програмуванні. 
На базовому рівні список являє собою впорядковану послідовність елементів, де кожен елемент має свій індекс або позицію. 
Існує дві основні концептуальні реалізації списків: динамічні масиви (Array-based Lists) та зв'язані списки (Linked Lists). 
У динамічних масивах елементи зберігаються у неперервній області пам'яті. Це забезпечує миттєвий доступ до будь-якого елемента за індексом за константний час O(1). 
Однак вставка або видалення елементів у середину чи початок динамічного масиву вимагає зсуву всіх наступних елементів, що займає лінійний час O(n). 
Зв'язані списки, у свою чергу, складаються з вузлів (nodes), де кожен вузол містить корисне значення та посилання (вказівник) на наступний або попередній елемент. 
Однозв'язні списки містять посилання лише в один бік, тоді як двозв'язні списки дозволяють навігацію в обох напрямках. 
Головною перевагою зв'язаних списків є швидке додавання та видалення елементів на початку чи в кінці за час O(1), проте доступ до довільного елемента за індексом вимагає послідовного обходу від початку і займає час O(n). 
Черга (Queue) — це спеціалізована лінійна структура даних, що працює за суворим принципом FIFO (First-In, First-Out: першим прийшов — першим пішов). 
У черзі нові елементи завжди додаються в кінець (операція Enqueue), а вилучаються лише з початку (операція Dequeue). 
Окрім цього, черги підтримують допоміжні операції, такі як перегляд першого елемента без його видалення (Peek або Front), а також перевірку на порожнечу (isEmpty). 
Черги відіграють критичну роль у системному програмуванні та мережевих архітектурах. 
Вони використовуються для обробки асинхронних завдань, керування чергами друку, розподілу ресурсів процесора (планування процесів), а також у алгоритмах обходу графів, зокрема у пошуку в ширину (BFS). 
У багатьох мовах програмування черги реалізуються на основі двозв'язаних списків або кільцевих буферів (Circular Buffers) для забезпечення константного часу O(1) для операцій Enqueue та Dequeue. 
Окремим різновидом є пріоритетна черга (Priority Queue), у якій кожен елемент має свій пріоритет, і елемент із найвищим пріоритетом вилучається першим, незалежно від часу його додавання. 
Стек (Stack) є дзеркальною за логікою до черги структурою даних, яка функціонує за принципом LIFO (Last-In, First-Out: останнім прийшов — першим пішов). 
Додавання (Push) та видалення (Pop) елементів у стек відбувається виключно з одного кінця, який називається верхівкою стека (Top). 
Стеки є фундаментом виконання програм на апаратному та системному рівнях: саме стек викликів (Call Stack) відстежує виконання функцій, зберігає локальні змінні та адреси повернення. 
Також стеки застосовуються для обробки рекурсивних викликів, перевірки балансу дужок у компіляторах, парсингу математичних виразів (обрана польська нотація) та реалізації функції скасування дій (Undo/Redo) у текстових редакторах. 
Порівнюючи ці структури, варто зазначити, що список є універсальним інструментом загального призначення з довільним доступом, тоді як черга та стек є обмеженими (дисциплінованими) структурами, що навмисно обмежують точки доступу до даних заради гарантії дотримання конкретного порядку обробки. 
Вибір між ними залежить від специфіки задачі: динамічні масиви кращі для читання та ітерації, зв'язані списки — для частої вставки на початку, черги — для потокової та впорядкованої обробки за часом, а стеки — для вкладених та зворотно-послідовних операцій.
"""

prompt_direct = f"Проаналізуй наданий текст та зроби його детальне та точно структуроване узагальнення:\n\n{large_text}"

interaction_direct = client.interactions.create(
    model=MODEL,
    input=prompt_direct
)

direct_result = interaction_direct.output_text.strip()

print("5.1 ОБРОБКА БЕЗ CHUNKING (DIRECT PROCESSING)")
print(direct_result)


words = large_text.split()
chunk_size = 120
chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

print(f"\n5.2 CHUNKING (MAP-REDUCE). Текст розділено на {len(chunks)} chunks")

summaries_chunking = []
for idx, chunk in enumerate(chunks, 1):
    prompt_map = f"Зроби коротке та точне узагальнення цього фрагмента тексту (Chunk {idx}):\n\n{chunk}"
    interaction_map = client.interactions.create(model=MODEL, input=prompt_map)
    summary = interaction_map.output_text.strip()
    summaries_chunking.append(summary)
    print(f"\nSummary for Chunk {idx}\n{summary}")

combined_chunking = "\n\n".join([f"Частина {i+1}: {s}" for i, s in enumerate(summaries_chunking)])
prompt_reduce_chunking = f"На основі наступних часткових узагальнень сформулюй один цілісний підсумковий висновок для всього документа:\n\n{combined_chunking}"

interaction_reduce_chunking = client.interactions.create(
    model=MODEL,
    input=prompt_reduce_chunking
)

chunking_result = interaction_reduce_chunking.output_text.strip()
print(f"\n>>> ФІНАЛЬНИЙ РЕЗУЛЬТАТ (MAP-REDUCE CHUNKING):\n{chunking_result}")


window_size = 120
overlap = 35
step = window_size - overlap
sliding_chunks = []

for i in range(0, len(words), step):
    chunk_words = words[i:i + window_size]
    sliding_chunks.append(" ".join(chunk_words))
    if i + window_size >= len(words):
        break

print(f"\nSLIDING WINDOW. Текст розділено на {len(sliding_chunks)} sliding chunks")

summaries_sliding = []
for idx, chunk in enumerate(sliding_chunks, 1):
    prompt_sliding = f"Зроби узагальнення цього фрагмента тексту з урахуванням суміжного контексту:\n\n{chunk}"
    interaction_sliding = client.interactions.create(model=MODEL, input=prompt_sliding)
    summary = interaction_sliding.output_text.strip()
    summaries_sliding.append(summary)
    print(f"\nSummary for Sliding Chunk {idx}\n{summary}")

combined_sliding = "\n\n".join([f"Фрагмент {i+1}: {s}" for i, s in enumerate(summaries_sliding)])
prompt_reduce_sliding = f"На основі наступних часткових узагальнень (зроблених із перекриттям контексту) сформулюй остаточне підсумкове узагальнення:\n\n{combined_sliding}"

interaction_reduce_sliding = client.interactions.create(
    model=MODEL,
    input=prompt_reduce_sliding
)

sliding_result = interaction_reduce_sliding.output_text.strip()
print(f"\n>>> ФІНАЛЬНИЙ РЕЗУЛЬТАТ (SLIDING WINDOW):\n{sliding_result}")


prompt_compare = f"""
Проаналізуй та порівняй три підходи до узагальнення великого тексту:

1. Пряма обробка без chunking (Direct):
{direct_result}

2. Метод Chunking (Map-Reduce без перекриття):
{chunking_result}

3. Метод Sliding Window (з перекриттям контексту):
{sliding_result}

Оціни їх за наступними критеріями:
- Повнота охоплення матеріалу та точність
- Збереження важливих фактів і деталей (складність O(1), O(n), види списків/черг)
- Структурованість підсумкового тексту
- Виявлення пропущеної інформації

Сформулюй чіткий висновок щодо доцільності використання кожного методу під час роботи з великими документами.
"""

interaction_compare = client.interactions.create(
    model=MODEL,
    input=prompt_compare
)

print("\n5.3 ПОРІВНЯЛЬНИЙ АНАЛІЗ РЕЗУЛЬТАТІВ")
print(interaction_compare.output_text.strip())