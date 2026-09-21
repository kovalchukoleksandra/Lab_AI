import gradio as gr
from PIL import Image


def show_image(image):
    if image is None:
        return "Будь ласка, завантажте зображення.", None

    return "Зображення успішно завантажено!", image


with gr.Blocks(title="Task 0 - Image Uploader") as demo:

    gr.Markdown("# Task 0 - Image Uploader")

    input_image = gr.Image(
        type="pil",
        label="Завантажте зображення"
    )

    button = gr.Button("Показати зображення")

    message = gr.Textbox(
        label="Повідомлення"
    )

    output_image = gr.Image(
        label="Завантажене зображення"
    )

    button.click(
        fn=show_image,
        inputs=input_image,
        outputs=[message, output_image]
    )


if __name__ == "__main__":
    demo.launch()