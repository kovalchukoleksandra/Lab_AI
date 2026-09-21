import gradio as gr
from PIL import Image, ImageDraw
import numpy as np


GRID_ROWS = 5
GRID_COLS = 5

def calculate_feature_vector(image):
    if image is None:
        return None, None, None

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

    return absolute_vector, normalized_vector, grid_image

def process_class(images, class_name):

    valid_images = [
        image for image in images
        if image is not None
    ]

    if len(valid_images) == 0:
        return (
            [],
            f"{class_name}: навчальні зразки не завантажені.",
            ""
        )

    gallery_images = []

    vectors_text = ""

    normalized_vectors = []

    for i, image in enumerate(valid_images, start=1):

        absolute_vector, normalized_vector, grid_image = \
            calculate_feature_vector(image)

        gallery_images.append(grid_image)

        normalized_vectors.append(normalized_vector)

        vectors_text += (
            f"\nЗРАЗОК {i}\n"
            f"Абсолютний вектор:\n"
            f"{absolute_vector}\n\n"
            f"Нормований вектор:\n"
            f"{normalized_vector}\n"
            f"\n{'-' * 60}\n"
        )

    center = np.mean(
        np.array(normalized_vectors),
        axis=0
    )

    center = [
        round(float(value), 4)
        for value in center
    ]

    statistics_text = (
        f"{class_name}\n\n"
        f"Кількість навчальних зразків: "
        f"{len(valid_images)}\n\n"
        f"Геометричний центр кластера:\n"
        f"{center}"
    )

    return (
        gallery_images,
        vectors_text,
        statistics_text
    )


def train_system(
    *images
):

    class_1_images = images[0:10]
    class_2_images = images[10:20]
    class_3_images = images[20:30]

    class_1_gallery, class_1_vectors, class_1_stats = \
        process_class(
            class_1_images,
            "КЛАС 1"
        )

    class_2_gallery, class_2_vectors, class_2_stats = \
        process_class(
            class_2_images,
            "КЛАС 2"
        )

    class_3_gallery, class_3_vectors, class_3_stats = \
        process_class(
            class_3_images,
            "КЛАС 3"
        )

    centers = []

    for class_images in [
        class_1_images,
        class_2_images,
        class_3_images
    ]:

        valid_images = [
            image for image in class_images
            if image is not None
        ]

        if len(valid_images) == 0:
            centers.append(None)
            continue

        normalized_vectors = []

        for image in valid_images:

            _, normalized_vector, _ = \
                calculate_feature_vector(image)

            normalized_vectors.append(
                normalized_vector
            )

        center = np.mean(
            np.array(normalized_vectors),
            axis=0
        )

        centers.append(center.tolist())

    status = (
        "НАВЧАННЯ ЗАВЕРШЕНО\n\n"
        "Система побудувала центри трьох кластерів.\n"
        "Для класифікації завантажте невідомий образ."
    )

    return (
        class_1_gallery,
        class_1_vectors,
        class_1_stats,

        class_2_gallery,
        class_2_vectors,
        class_2_stats,

        class_3_gallery,
        class_3_vectors,
        class_3_stats,

        centers,
        status
    )


def classify_unknown(image, centers):

    if image is None:
        return (
            "Невідоме зображення не завантажено.",
            "",
            None,
            ""
        )

    if centers is None:
        return (
            "Спочатку потрібно виконати навчання системи.",
            "",
            None,
            ""
        )

    if any(center is None for center in centers):
        return (
            "Для всіх трьох класів потрібно завантажити "
            "навчальні зразки.",
            "",
            None,
            ""
        )

    absolute_vector, normalized_vector, grid_image = \
        calculate_feature_vector(image)

    distances = []

    for center in centers:

        distance = np.sqrt(
            np.sum(
                (
                    np.array(normalized_vector)
                    - np.array(center)
                ) ** 2
            )
        )

        distances.append(float(distance))

    predicted_class = (
        int(np.argmin(distances)) + 1
    )

    distances_text = (
        f"Відстань до Класу 1: "
        f"{distances[0]:.4f}\n"
        f"Відстань до Класу 2: "
        f"{distances[1]:.4f}\n"
        f"Відстань до Класу 3: "
        f"{distances[2]:.4f}"
    )

    result_text = (
        f"РОЗПІЗНАВАННЯ\n\n"
        f"Найменша відстань: "
        f"{distances[predicted_class - 1]:.4f}\n\n"
        f"Невідомий образ належить до "
        f"Класу {predicted_class}."
    )

    absolute_text = (
        "Абсолютний вектор ознак:\n\n"
        f"{absolute_vector}"
    )

    normalized_text = (
        "Нормований вектор ознак:\n\n"
        f"{normalized_vector}"
    )

    return (
        absolute_text,
        normalized_text,
        grid_image,
        distances_text + "\n\n" + result_text
    )

with gr.Blocks(
    title="Система розпізнавання з навчанням"
) as demo:

    gr.Markdown(
        """
        # Система розпізнавання образів з навчанням

        **Варіант 1 — геометричні центри кластерів**

        Для кожного з трьох класів використовується
        навчальна послідовність із 10 зображень.
        """
    )

    gr.Markdown("## Клас 1 — навчальні зразки")

    class_1_inputs = []

    with gr.Row():

        for i in range(10):

            image = gr.Image(
                type="pil",
                label=f"Зразок 1.{i + 1}"
            )

            class_1_inputs.append(image)

    class_1_gallery = gr.Gallery(
        label="Клас 1 — зображення із сіткою 5×5",
        columns=5,
        height="auto"
    )

    class_1_vectors = gr.Textbox(
        label="Клас 1 — абсолютні та нормовані вектори",
        lines=20
    )

    class_1_stats = gr.Textbox(
        label="Клас 1 — статистика кластера",
        lines=5
    )

    gr.Markdown("## Клас 2 — навчальні зразки")

    class_2_inputs = []

    with gr.Row():

        for i in range(10):

            image = gr.Image(
                type="pil",
                label=f"Зразок 2.{i + 1}"
            )

            class_2_inputs.append(image)

    class_2_gallery = gr.Gallery(
        label="Клас 2 — зображення із сіткою 5×5",
        columns=5,
        height="auto"
    )

    class_2_vectors = gr.Textbox(
        label="Клас 2 — абсолютні та нормовані вектори",
        lines=20
    )

    class_2_stats = gr.Textbox(
        label="Клас 2 — статистика кластера",
        lines=5
    )

    gr.Markdown("## Клас 3 — навчальні зразки")

    class_3_inputs = []

    with gr.Row():

        for i in range(10):

            image = gr.Image(
                type="pil",
                label=f"Зразок 3.{i + 1}"
            )

            class_3_inputs.append(image)

    class_3_gallery = gr.Gallery(
        label="Клас 3 — зображення із сіткою 5×5",
        columns=5,
        height="auto"
    )

    class_3_vectors = gr.Textbox(
        label="Клас 3 — абсолютні та нормовані вектори",
        lines=20
    )

    class_3_stats = gr.Textbox(
        label="Клас 3 — статистика кластера",
        lines=5
    )

    train_button = gr.Button(
        "Навчити систему",
        variant="primary"
    )

    training_status = gr.Textbox(
        label="Стан навчання",
        lines=5
    )

    centers_state = gr.State(None)

    gr.Markdown("## Розпізнавання невідомого образу")

    unknown_image = gr.Image(
        type="pil",
        label="Завантажте невідоме зображення"
    )

    classify_button = gr.Button(
        "Розпізнати невідомий образ",
        variant="primary"
    )

    with gr.Row():

        unknown_absolute = gr.Textbox(
            label="Абсолютний вектор",
            lines=8
        )

        unknown_normalized = gr.Textbox(
            label="Нормований вектор",
            lines=8
        )

    unknown_grid = gr.Image(
        label="Невідомий образ із сіткою 5×5"
    )

    classification_result = gr.Textbox(
        label="Результат класифікації",
        lines=8
    )

    train_button.click(
        fn=train_system,
        inputs=(
            class_1_inputs
            + class_2_inputs
            + class_3_inputs
        ),
        outputs=[
            class_1_gallery,
            class_1_vectors,
            class_1_stats,

            class_2_gallery,
            class_2_vectors,
            class_2_stats,

            class_3_gallery,
            class_3_vectors,
            class_3_stats,

            centers_state,
            training_status
        ]
    )

  
    classify_button.click(
        fn=classify_unknown,
        inputs=[
            unknown_image,
            centers_state
        ],
        outputs=[
            unknown_absolute,
            unknown_normalized,
            unknown_grid,
            classification_result
        ]
    )


if __name__ == "__main__":
    demo.launch()