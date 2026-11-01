import random
import numpy as np
from PIL import Image, ImageDraw

GRID_ROWS = 5
GRID_COLS = 5

R_DEFAULT = 0.5
MAX_EPOCHS = 100

CLASS_1 = -1
CLASS_2 = 1


def calculate_feature_vector(image):
    if image is None:
        return None, None, None

    gray_image = image.convert("L")
    width, height = gray_image.size

    pixels = np.array(gray_image)

    cell_width = width / GRID_COLS
    cell_height = height / GRID_ROWS

    absolute_vector = []

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):

            x1 = int(col * cell_width)
            x2 = int((col + 1) * cell_width)

            y1 = int(row * cell_height)
            y2 = int((row + 1) * cell_height)

            cell = pixels[y1:y2, x1:x2]

            black_pixels = int(np.sum(cell < 128))

            absolute_vector.append(black_pixels)

    total_black_pixels = sum(absolute_vector)

    if total_black_pixels > 0:
        normalized_vector = [
            value / total_black_pixels
            for value in absolute_vector
        ]
    else:
        normalized_vector = [0.0] * (GRID_ROWS * GRID_COLS)

    grid_image = gray_image.convert("RGB").copy()
    draw = ImageDraw.Draw(grid_image)

    for col in range(1, GRID_COLS):
        x = int(col * cell_width)
        draw.line((x, 0, x, height), fill="red", width=2)

    for row in range(1, GRID_ROWS):
        y = int(row * cell_height)
        draw.line((0, y, width, y), fill="red", width=2)

    return absolute_vector, normalized_vector, grid_image


def create_perceptron():
    weights = [
        random.uniform(0 , 1)
        for _ in range(GRID_ROWS * GRID_COLS)
    ]
    b = random.uniform(0 , 1)

    return weights, b


def predict(normalized_vector, weights, b):
    x = np.array(normalized_vector, dtype=float)
    w = np.array(weights, dtype=float)
    weighted_sum = float(np.dot(x, w) + b)

    if weighted_sum >= 0:
        output = CLASS_2
    else:
        output = CLASS_1

    return weighted_sum, output


def get_class_name(output):
    if output == CLASS_1:
        return "Клас 1"
    return "Клас 2"


def update_perceptron(
    normalized_vector,
    desired_output,
    weights,
    b,
    learning_rate
):

    weighted_sum, output = predict(
        normalized_vector, weights, b
    )

    error = desired_output - output

    if output == desired_output:
        return weights, b, weighted_sum, output, error

    x = np.array(normalized_vector, dtype=float)
    w = np.array(weights, dtype=float)

    w = w + learning_rate * error * x
    b = b + learning_rate * error

    weights = w.tolist()

    return weights, b, weighted_sum, output, error


def format_weights(weights):
    if weights is None:
        return "Перцептрон ще не створено."

    return "\n".join(
        f"w{i} = {weight:.6f}"
        for i, weight in enumerate(weights, start=1)
    )


def format_vectors(absolute_vector, normalized_vector):
    normalized_text = [
        round(value, 6)
        for value in normalized_vector
    ]

    return (
        f"Абсолютний вектор ознак ({len(absolute_vector)} значень):\n"
        f"{absolute_vector}\n\n"
        "----------------------------------------\n\n"
        f"Нормований вектор ознак ({len(normalized_vector)} значень):\n"
        f"{normalized_text}\n\n"
        f"Сума нормованих ознак: "
        f"{sum(normalized_vector):.6f}"
    )


def initialize_system(learning_rate):
    if learning_rate is None or learning_rate <= 0:
        learning_rate = R_DEFAULT

    weights, b = create_perceptron()

    status = (
        "ПЕРЦЕПТРОН СТВОРЕНО\n\n"
        f"Кількість входів: {len(weights)}\n"
        f"Кількість ваг: {len(weights)}\n"
        f"Швидкість навчання R: {learning_rate}\n"
        f"Початкове зміщення b: {b:.6f}\n"
        "Навчальних прикладів поки немає."
    )

    return (
        weights,
        b,
        0,
        [],
        format_weights(weights),
        f"{b:.6f}",
        0,
        status,
        "",
        None,
        "",
        "",
        "",
    )


def recognize_image(image, weights, b):
    if image is None:
        return (
            "Завантажте зображення.",
            None,
            "Розпізнавання неможливе."
        )

    if weights is None or b is None:
        return (
            "Спочатку створіть перцептрон.",
            None,
            "Перцептрон ще не ініціалізовано."
        )

    absolute_vector, normalized_vector, grid_image = (
        calculate_feature_vector(image)
    )

    weighted_sum, output = predict(
        normalized_vector, weights, b
    )

    vectors_text = format_vectors(
        absolute_vector, normalized_vector
    )

    result_text = (
        f"Зважена сума S = {weighted_sum:.6f}\n"
        f"Зміщення b = {b:.6f}\n\n"
        f"Вихід перцептрона y = {output}\n"
        f"Розпізнаний клас: {get_class_name(output)}"
    )

    return vectors_text, grid_image, result_text


def train_on_image(
    image,
    correct_class,
    weights,
    b,
    step,
    training_samples,
    learning_rate
):
    if image is None:
        return (
            weights, b, step, training_samples,
            "Завантажте зображення.",
            format_weights(weights),
            f"{b:.6f}" if b is not None else "",
            step, "", None
        )

    if weights is None or b is None:
        return (
            weights, b, step, training_samples,
            "Спочатку створіть перцептрон.",
            format_weights(weights),
            "",
            step, "", None
        )

    if learning_rate is None or learning_rate <= 0:
        learning_rate = R_DEFAULT

    if training_samples is None:
        training_samples = []

    absolute_vector, normalized_vector, grid_image = (
        calculate_feature_vector(image)
    )

    desired_output = (
        CLASS_1 if correct_class == "Клас 1" else CLASS_2
    )

    weighted_sum, output = predict(
        normalized_vector, weights, b
    )

    error = desired_output - output

    already_saved = any(
        sample["x"] == normalized_vector
        and sample["target"] == desired_output
        for sample in training_samples
    )

    if not already_saved:
        training_samples.append({
            "x": normalized_vector.copy(),
            "target": desired_output,
            "label": correct_class,
        })

    step += 1

    if output == desired_output:
        result_text = (
            "КЛАСИФІКАЦІЯ ПРАВИЛЬНА\n\n"
            f"Розпізнаний клас: {get_class_name(output)}\n"
            f"Правильний клас: {correct_class}\n"
            f"Вихід y = {output}\n"
            f"Бажаний вихід d = {desired_output}\n"
            "Помилка класифікації відсутня.\n"
            "Ваги та зміщення не змінюються."
        )

    else:
        weights, b, _, _, _ = update_perceptron(
            normalized_vector,
            desired_output,
            weights,
            b,
            learning_rate,
        )

        new_sum, new_output = predict(
            normalized_vector, weights, b
        )

        result_text = (
            "КЛАСИФІКАЦІЯ БУЛА НЕПРАВИЛЬНОЮ\n\n"
            f"Попередній вихід y = {output}\n"
            f"Правильний вихід d = {desired_output}\n"
            f"Помилка d - y = {error}\n\n"
            "Ваги та зміщення скориговано.\n\n"
            f"Нова зважена сума S = {new_sum:.6f}\n"
            f"Новий вихід y = {new_output}\n"
            f"Клас після корекції: {get_class_name(new_output)}\n"
            f"Правильний клас: {correct_class}\n"
            f"Результат після одного оновлення: "
            f"{'правильний' if new_output == desired_output else 'ще неправильний'}"
        )

    result_text += (
        f"\n\nКрок навчання: {step}"
        f"\nШвидкість навчання R = {learning_rate}"
        f"\nЗбережено навчальних прикладів: {len(training_samples)}"
        f"\nЗміщення b = {b:.6f}"
    )

    vectors_text = format_vectors(
        absolute_vector, normalized_vector
    )

    return (
        weights,
        b,
        step,
        training_samples,
        result_text,
        format_weights(weights),
        f"{b:.6f}",
        step,
        vectors_text,
        grid_image,
    )


def check_convergence(
    weights,
    b,
    step,
    training_samples,
    learning_rate
):
    if weights is None or b is None:
        return (
            weights, b, step,
            format_weights(weights), "",
            step,
            "Перцептрон не створено.",
            "Спочатку створіть перцептрон."
        )

    if not training_samples:
        return (
            weights, b, step,
            format_weights(weights), f"{b:.6f}",
            step,
            "Навчальний набір порожній.",
            "Додайте приклади обох класів і повторіть перевірку."
        )

    if learning_rate is None or learning_rate <= 0:
        learning_rate = R_DEFAULT

    initial_weights = weights.copy()
    initial_b = b

    total_samples = len(training_samples)
    epochs_used = 0
    total_mistakes = total_samples

    for epoch in range(1, MAX_EPOCHS + 1):

        for sample in training_samples:
            weights, b, _, _, _ = update_perceptron(
                sample["x"],
                sample["target"],
                weights,
                b,
                learning_rate,
            )
            step += 1

        total_mistakes = 0

        for sample in training_samples:
            _, output = predict(
                sample["x"], weights, b
            )

            if output != sample["target"]:
                total_mistakes += 1

        epochs_used = epoch

        if total_mistakes == 0:
            break

    correct = total_samples - total_mistakes
    accuracy = correct / total_samples * 100

    converged = total_mistakes == 0

    if converged:
        conclusion = (
            "ЗБІЖНІСТЬ ДОСЯГНУТО.\n"
            "Усі збережені навчальні приклади "
            "класифікуються правильно."
        )
    else:
        conclusion = (
            f"ЗБІЖНІСТЬ НЕ ДОСЯГНУТО за {MAX_EPOCHS} епох.\n"
            "У навчальному наборі залишилися помилки. "
            "Перевірте приклади та їхні правильні класи."
        )

    weights_changed = (
        weights != initial_weights or b != initial_b
    )

    result_text = (
        f"Кількість навчальних прикладів: {total_samples}\n"
        f"Кількість виконаних епох: {epochs_used}\n"
        f"Загальний лічильник кроків: {step}\n"
        f"Правильних класифікацій: {correct}\n"
        f"Помилок після останньої епохи: {total_mistakes}\n"
        f"Точність на навчальному наборі: {accuracy:.2f}%\n"
        f"Швидкість навчання R: {learning_rate}\n\n"
        f"{conclusion}\n\n"
        f"Початкове зміщення b: {initial_b:.6f}\n"
        f"Поточне зміщення b: {b:.6f}\n"
        f"Параметри змінилися: {'так' if weights_changed else 'ні'}"
    )

    status = (
        f"Перевірено {total_samples} навчальних прикладів. "
        f"Виконано епох: {epochs_used}. "
        f"Залишилося помилок: {total_mistakes}."
    )

    return (
        weights,
        b,
        step,
        format_weights(weights),
        f"{b:.6f}",
        step,
        status,
        result_text,
    )


def reset_system():
    return (
        None,
        None,
        0,
        [],
        "",
        "",
        0,
        "Систему скинуто. Створіть новий перцептрон.",
        "",
        None,
        "",
        "",
        "",
    )