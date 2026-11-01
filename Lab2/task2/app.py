import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

GRID_SIZE = 5
N_FEATURES = GRID_SIZE * GRID_SIZE
N_CLASSES = 3
MAX_ITERATIONS = 100

def image_to_vector(image):
    if image is None:
        raise ValueError("Завантажте всі чотири зображення.")

    gray_image = image.convert("L")
    pixels = np.array(gray_image)

    height, width = pixels.shape
    cell_width = width / GRID_SIZE
    cell_height = height / GRID_SIZE

    absolute_vector = []
    bipolar_vector = []

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x1 = int(col * cell_width)
            x2 = int((col + 1) * cell_width)
            y1 = int(row * cell_height)
            y2 = int((row + 1) * cell_height)

            cell = pixels[y1:y2, x1:x2]

            black_pixels = int(np.sum(cell < 128))
            total_pixels = cell.size

            absolute_vector.append(black_pixels)

            if black_pixels > total_pixels / 2:
                bipolar_vector.append(1)
            else:
                bipolar_vector.append(-1)

    grid_image = gray_image.convert("RGB").copy()
    draw = ImageDraw.Draw(grid_image)

    for col in range(1, GRID_SIZE):
        x = int(col * cell_width)
        draw.line((x, 0, x, height), fill="red", width=2)

    for row in range(1, GRID_SIZE):
        y = int(row * cell_height)
        draw.line((0, y, width, y), fill="red", width=2)

    return (
        np.array(absolute_vector),
        np.array(bipolar_vector),
        grid_image
    )

def create_hopfield_weights(patterns):
    n = len(patterns[0])
    weights = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(len(patterns)):
                    weights[i][j] += patterns[k][i] * patterns[k][j]

                weights[i][j] /= n

    return weights


def hopfield_recognize(input_vector, weights):

    y = input_vector.copy()

    for iteration in range(MAX_ITERATIONS):
        previous = y.copy()

        for j in range(len(y)):
            activation = np.dot(weights[j], y)

            if activation > 0:
                y[j] = 1
            elif activation < 0:
                y[j] = -1

        if np.array_equal(y, previous):
            return y, iteration + 1, True

    return y, MAX_ITERATIONS, False


def create_hamming_weights(patterns):
    n = patterns.shape[1]

    weights = patterns.T / 2

    biases = np.full(patterns.shape[0], n / 2)

    return weights, biases


def hamming_recognize(input_vector, weights, biases):
    scores = input_vector @ weights + biases

    outputs = np.maximum(scores, 0).astype(float)

    epsilon = 1 / (N_CLASSES + 1)

    for iteration in range(1000):
        previous = outputs.copy()

        for j in range(N_CLASSES):
            other_sum = np.sum(outputs) - outputs[j]

            outputs[j] = max(
                0.0,
                outputs[j] - epsilon * other_sum
            )

        if np.allclose(outputs, previous, atol=1e-10):
            break

    winner = int(np.argmax(scores))

    return scores, outputs, winner, iteration + 1

def format_vector(vector):
    return np.array2string(
        vector,
        separator=", ",
        max_line_width=120
    )


def format_matrix(matrix):
    return np.array2string(
        matrix,
        separator=" ",
        precision=2,
        suppress_small=True,
        max_line_width=140
    )


def recognize(image1, image2, image3, unknown, network):
    images = [image1, image2, image3, unknown]

    if any(image is None for image in images):
        return (
            None, None, None, None,
            "Завантажте три еталонні зображення та невідоме зображення.",
            "",
            ""
        )

    try:
        processed = [image_to_vector(image) for image in images]

        absolute_vectors = np.array([item[0] for item in processed])
        bipolar_vectors = np.array([item[1] for item in processed])
        previews = [item[2] for item in processed]

        reference_patterns = bipolar_vectors[:3]
        unknown_vector = bipolar_vectors[3]

        vector_text = (
            "Абсолютні вектори (кількість чорних пікселів):\n"
            + "\n".join(
                f"Зображення {i + 1}: {format_vector(vector)}"
                for i, vector in enumerate(absolute_vectors)
            )
            + "\n\nБіполярні вектори (-1 та 1):\n"
            + "\n".join(
                f"Зображення {i + 1}: {format_vector(vector)}"
                for i, vector in enumerate(bipolar_vectors)
            )
        )

        if network == "Хопфілд":
            weights = create_hopfield_weights(reference_patterns)

            result, iterations, converged = hopfield_recognize(
                unknown_vector, weights
            )

            distances = np.sum(
                reference_patterns != result, axis=1
            )
            winner = int(np.argmin(distances))

            matrix_text = (
                "Матриця ваг мережі Хопфілда:\n"
                + format_matrix(weights)
            )

            result_text = (
                f"Обрана мережа: {network}\n"
                f"Результат: клас {winner + 1}\n"
                f"Кількість ітерацій: {iterations}\n"
                f"Збіжність: {'так' if converged else 'ні'}\n"
                f"Відстані до еталонів: {format_vector(distances)}\n"
                f"Відновлений вектор: {format_vector(result)}"
            )

            if distances[winner] != 0:
                result_text += (
                    "\nУвага: відновлений образ не збігається "
                    "повністю з жодним еталоном."
                )

        else:
            weights, biases = create_hamming_weights(
                reference_patterns
            )

            scores, outputs, winner, iterations = hamming_recognize(
                unknown_vector, weights, biases
            )

            matrix_text = (
                "Ваги першого шару мережі Хеммінга:\n"
                + format_matrix(weights)
                + "\n\nЗміщення:\n"
                + format_vector(biases)
            )

            result_text = (
                f"Обрана мережа: {network}\n"
                f"Результат: клас {winner + 1}\n"
                f"Оцінки схожості: {format_vector(scores)}\n"
                f"Виходи конкурентного шару: {format_vector(outputs)}\n"
                f"Кількість ітерацій: {iterations}"
            )

            if np.sum(np.isclose(scores, np.max(scores))) > 1:
                result_text += (
                    "\nУвага: кілька класів мають однакову "
                    "оцінку схожості."
                )

        return (
            previews[0],
            previews[1],
            previews[2],
            previews[3],
            vector_text,
            matrix_text,
            result_text
        )

    except Exception as error:
        return (
            None, None, None, None,
            f"Помилка обробки: {error}",
            "",
            ""
        )

with gr.Blocks(title="Розпізнавання образів") as demo:

    gr.Markdown(
        "# Лабораторна робота № 2\n"
        "## Нейронні мережі Хопфілда та Хеммінга\n"
        "Завантажте три еталони та невідоме зображення. "
        "Для коректного порівняння використовуйте однакові "
        "за стилем символи з контрастними контурами."
    )

    with gr.Row():
        image1 = gr.Image(
            label="Еталон класу 1",
            type="pil"
        )
        image2 = gr.Image(
            label="Еталон класу 2",
            type="pil"
        )

    with gr.Row():
        image3 = gr.Image(
            label="Еталон класу 3",
            type="pil"
        )
        unknown = gr.Image(
            label="Невідоме зображення",
            type="pil"
        )

    network = gr.Radio(
        choices=["Хопфілд", "Хеммінг"],
        value="Хопфілд",
        label="Оберіть нейронну мережу"
    )

    button = gr.Button(
        "Розпізнати зображення",
        variant="primary"
    )

    gr.Markdown("### Оброблені зображення (5 × 5)")

    with gr.Row():
        preview1 = gr.Image(label="Клас 1", interactive=False)
        preview2 = gr.Image(label="Клас 2", interactive=False)
        preview3 = gr.Image(label="Клас 3", interactive=False)
        preview_unknown = gr.Image(
            label="Невідомий образ",
            interactive=False
        )

    vector_output = gr.Textbox(
        label="Абсолютні та біполярні вектори",
        lines=12
    )

    matrix_output = gr.Textbox(
        label="Матриця ваг або ваги першого шару",
        lines=15
    )

    result_output = gr.Textbox(
        label="Результат класифікації",
        lines=10
    )

    button.click(
        fn=recognize,
        inputs=[
            image1,
            image2,
            image3,
            unknown,
            network
        ],
        outputs=[
            preview1,
            preview2,
            preview3,
            preview_unknown,
            vector_output,
            matrix_output,
            result_output
        ]
    )


if __name__ == "__main__":
    demo.launch()
