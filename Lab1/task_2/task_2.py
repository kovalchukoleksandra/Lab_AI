import gradio as gr
from PIL import Image, ImageDraw
import numpy as np

GRID_ROWS = 5
GRID_COLS = 5

def calculate_feature_vector(image):
    if image is None:
        return [], []

    image = image.convert("L")
    width, height = image.size
    pixels = np.array(image)
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
            black_pixels = np.sum(cell < 128)

            absolute_vector.append(int(black_pixels))

    total_black_pixels = sum(absolute_vector)

    if total_black_pixels > 0:
        normalized_vector = [
            round(value / total_black_pixels, 4)
            for value in absolute_vector
        ]
    else:
        normalized_vector = [0.0] * len(absolute_vector)

    return absolute_vector, normalized_vector

def create_grid_image(image):
    if image is None:
        return None

    image = image.convert("L")

    width, height = image.size

    cell_width = width / GRID_COLS
    cell_height = height / GRID_ROWS

    grid_image = image.convert("RGB").copy()

    draw = ImageDraw.Draw(grid_image)

    for col in range(1, GRID_COLS):
        x = int(col * cell_width)

        draw.line(
            (x, 0, x, height),
            fill="red",
            width=2
        )

    for row in range(1, GRID_ROWS):
        y = int(row * cell_height)

        draw.line(
            (0, y, width, y),
            fill="red",
            width=2
        )

    return grid_image

# Чебишевська відстань
def chebyshev_distance(vector_1, vector_2):
    differences = [
        abs(a - b)
        for a, b in zip(vector_1, vector_2)
    ]

    return max(differences)


def process_reference(image, class_name):
    if image is None:
        return (
            f"{class_name}: зображення не завантажено.",
            "",
            None
        )

    absolute_vector, normalized_vector = calculate_feature_vector(image)

    grid_image = create_grid_image(image)

    absolute_text = (
        f"{class_name}\n\n"
        f"Абсолютний вектор:\n"
        f"{absolute_vector}"
    )

    normalized_text = (
        f"{class_name}\n\n"
        f"Нормований вектор:\n"
        f"{normalized_vector}"
    )

    return absolute_text, normalized_text, grid_image


def recognize_image(
    reference_1,
    reference_2,
    reference_3,
    reference_4,
    unknown_image
):

    references = [
        reference_1,
        reference_2,
        reference_3,
        reference_4
    ]

    for i, reference in enumerate(references, start=1):
        if reference is None:
            return (
                "Помилка",
                "Будь ласка, завантажте всі 4 еталонні зображення.",
                "",
                None
            )

    if unknown_image is None:
        return (
            "Помилка",
            "Будь ласка, завантажте невідоме зображення.",
            "",
            None
        )

    reference_vectors = []

    for reference in references:
        absolute_vector, normalized_vector = (
            calculate_feature_vector(reference)
        )

        reference_vectors.append(
            (absolute_vector, normalized_vector)
        )

    unknown_absolute, unknown_normalized = (
        calculate_feature_vector(unknown_image)
    )

  
    distances = []

    for _, reference_normalized in reference_vectors:

        distance = chebyshev_distance(
            unknown_normalized,
            reference_normalized
        )

        distances.append(distance)

    nearest_class = distances.index(min(distances)) + 1

    min_distance = min(distances)

   
    distance_text = "Чебишевські відстані:\n\n"

    for i, distance in enumerate(distances, start=1):
        distance_text += (
            f"Клас {i}: {distance:.4f}\n"
        )

    result_text = (
        f"Результат класифікації:\n\n"
        f"Невідомий образ належить до Класу {nearest_class}.\n\n"
        f"Найменша відстань: {min_distance:.4f}"
    )

    unknown_grid = create_grid_image(unknown_image)

    unknown_text = (
        "Абсолютний вектор невідомого образу:\n\n"
        f"{unknown_absolute}\n\n\n"
        "Нормований вектор невідомого образу:\n\n"
        f"{unknown_normalized}"
    )

    return (
        result_text,
        distance_text,
        unknown_text,
        unknown_grid
    )



with gr.Blocks(
    title="Система розпізнавання образів"
) as demo:

    gr.Markdown(
        """
        # Система розпізнавання образів

        Метод порівняння з еталоном.

        Для порівняння використовується
        **Чебишевська відстань**.
        """
    )

    gr.Markdown("## Еталонні образи")

    with gr.Row():

        with gr.Column():

            reference_1 = gr.Image(
                type="pil",
                label="Еталон — Клас 1"
            )

            reference_1_grid = gr.Image(
                label="Сітка 5×5 — Клас 1"
            )

            reference_1_absolute = gr.Textbox(
                label="Абсолютний вектор — Клас 1",
                lines=4
            )

            reference_1_normalized = gr.Textbox(
                label="Нормований вектор — Клас 1",
                lines=4
            )

        with gr.Column():

            reference_2 = gr.Image(
                type="pil",
                label="Еталон — Клас 2"
            )

            reference_2_grid = gr.Image(
                label="Сітка 5×5 — Клас 2"
            )

            reference_2_absolute = gr.Textbox(
                label="Абсолютний вектор — Клас 2",
                lines=4
            )

            reference_2_normalized = gr.Textbox(
                label="Нормований вектор — Клас 2",
                lines=4
            )

    with gr.Row():

        with gr.Column():

            reference_3 = gr.Image(
                type="pil",
                label="Еталон — Клас 3"
            )

            reference_3_grid = gr.Image(
                label="Сітка 5×5 — Клас 3"
            )

            reference_3_absolute = gr.Textbox(
                label="Абсолютний вектор — Клас 3",
                lines=4
            )

            reference_3_normalized = gr.Textbox(
                label="Нормований вектор — Клас 3",
                lines=4
            )

        with gr.Column():

            reference_4 = gr.Image(
                type="pil",
                label="Еталон — Клас 4"
            )

            reference_4_grid = gr.Image(
                label="Сітка 5×5 — Клас 4"
            )

            reference_4_absolute = gr.Textbox(
                label="Абсолютний вектор — Клас 4",
                lines=4
            )

            reference_4_normalized = gr.Textbox(
                label="Нормований вектор — Клас 4",
                lines=4
            )

    process_references_button = gr.Button(
        "Обробити еталонні образи",
        variant="primary"
    )

    process_references_button.click(
        fn=lambda r1, r2, r3, r4: (
            *process_reference(r1, "Клас 1"),
            *process_reference(r2, "Клас 2"),
            *process_reference(r3, "Клас 3"),
            *process_reference(r4, "Клас 4")
        ),
        inputs=[
            reference_1,
            reference_2,
            reference_3,
            reference_4
        ],
        outputs=[
            reference_1_absolute,
            reference_1_normalized,
            reference_1_grid,

            reference_2_absolute,
            reference_2_normalized,
            reference_2_grid,

            reference_3_absolute,
            reference_3_normalized,
            reference_3_grid,

            reference_4_absolute,
            reference_4_normalized,
            reference_4_grid
        ]
    )

    gr.Markdown("## Невідомий образ")

    unknown_image = gr.Image(
        type="pil",
        label="Завантажте невідоме зображення"
    )

    recognize_button = gr.Button(
        "Розпізнати образ",
        variant="primary"
    )

    unknown_grid = gr.Image(
        label="Невідомий образ із сіткою 5×5"
    )

    unknown_vectors = gr.Textbox(
        label="Вектори ознак невідомого образу",
        lines=10
    )

  
    gr.Markdown("## Результати розпізнавання")

    classification_result = gr.Textbox(
        label="Класифікація",
        lines=4
    )

    distances_result = gr.Textbox(
        label="Міри відповідності",
        lines=6
    )

    recognize_button.click(
        fn=recognize_image,
        inputs=[
            reference_1,
            reference_2,
            reference_3,
            reference_4,
            unknown_image
        ],
        outputs=[
            classification_result,
            distances_result,
            unknown_vectors,
            unknown_grid
        ]
    )

if __name__ == "__main__":
    demo.launch()