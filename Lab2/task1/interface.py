import gradio as gr

from app import (
    R_DEFAULT,
    initialize_system,
    recognize_image,
    train_on_image,
    check_convergence,
    reset_system,
)


with gr.Blocks(title="Перцептрон Розенблата") as demo:

    gr.Markdown(
        """
        # Система розпізнавання перцептронного типу

        Розпізнавання образів двох класів за допомогою
        перцептрона Розенблата.

        Зображення поділяється на сітку 5×5.
        Для кожного зображення обчислюються 25 абсолютних
        та 25 нормованих ознак.
        """
    )

    weights_state = gr.State(None)
    b_state = gr.State(None)
    step_state = gr.State(0)
    training_samples_state = gr.State([])

    gr.Markdown("## 1. Початкові параметри")

    learning_rate = gr.Number(
        label="Швидкість навчання R",
        value=R_DEFAULT,
        minimum=0.01,
        maximum=1.0,
        step=0.05,
    )

    with gr.Row():
        initialize_button = gr.Button(
            "Створити перцептрон",
            variant="primary"
        )
        reset_button = gr.Button("Скинути систему")

    initial_status = gr.Textbox(
        label="Стан системи",
        lines=4,
    )

    with gr.Row():
        b_output = gr.Textbox(
            label="Поточне зміщення b"
        )
        step_output = gr.Number(
            label="Крок навчання",
            value=0,
            interactive=False,
        )

    weights_output = gr.Textbox(
        label="Поточні ваги перцептрона",
        lines=10,
    )

    gr.Markdown("## 2. Завантаження та розпізнавання образу")

    training_image = gr.Image(
        type="pil",
        label="Завантажте зображення",
    )

    correct_class = gr.Radio(
        choices=["Клас 1", "Клас 2"],
        value="Клас 1",
        label="Правильний клас зображення",
    )

    with gr.Row():
        recognize_button = gr.Button("Розпізнати образ")
        train_button = gr.Button(
            "Підтвердити / скоригувати",
            variant="primary",
        )

    vectors_output = gr.Textbox(
        label="Абсолютний і нормований вектори ознак",
        lines=8,
    )

    grid_output = gr.Image(
        label="Зображення із сіткою 5×5",
    )

    recognition_output = gr.Textbox(
        label="Результат розпізнавання",
        lines=5,
    )

    training_output = gr.Textbox(
        label="Результат навчання та корекції",
        lines=10,
    )

    gr.Markdown("## 3. Оцінювання збіжності навчання")

    gr.Markdown(
        """
        Спочатку послідовно додайте навчальні зразки,
        указуючи правильний клас кожного зображення.
        Після цього запустіть перевірку збіжності.
        """
    )

    convergence_button = gr.Button(
        "Перевірити збіжність",
        variant="primary",
    )

    convergence_output = gr.Textbox(
        label="Результат оцінювання збіжності",
        lines=10,
    )

    initialize_button.click(
        fn=initialize_system,
        inputs=[learning_rate],
        outputs=[
            weights_state,
            b_state,
            step_state,
            training_samples_state,
            weights_output,
            b_output,
            step_output,
            initial_status,
            vectors_output,
            grid_output,
            recognition_output,
            training_output,
            convergence_output,
        ],
    )

    recognize_button.click(
        fn=recognize_image,
        inputs=[
            training_image,
            weights_state,
            b_state,
        ],
        outputs=[
            vectors_output,
            grid_output,
            recognition_output,
        ],
    )

    train_button.click(
        fn=train_on_image,
        inputs=[
            training_image,
            correct_class,
            weights_state,
            b_state,
            step_state,
            training_samples_state,
            learning_rate,
        ],
        outputs=[
            weights_state,
            b_state,
            step_state,
            training_samples_state,
            training_output,
            weights_output,
            b_output,
            step_output,
            vectors_output,
            grid_output,
        ],
    )

    convergence_button.click(
        fn=check_convergence,
        inputs=[
            weights_state,
            b_state,
            step_state,
            training_samples_state,
            learning_rate,
        ],
        outputs=[
            weights_state,
            b_state,
            step_state,
            weights_output,
            b_output,
            step_output,
            initial_status,
            convergence_output,
        ],
    )

    reset_button.click(
        fn=reset_system,
        inputs=[],
        outputs=[
            weights_state,
            b_state,
            step_state,
            training_samples_state,
            weights_output,
            b_output,
            step_output,
            initial_status,
            vectors_output,
            grid_output,
            recognition_output,
            training_output,
            convergence_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()