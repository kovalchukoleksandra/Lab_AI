from google import genai

client = genai.Client(api_key="API_KEY")
MODEL = "gemini-3.5-flash-lite"

prompt_creative = "Придумай 3 оригінальні назви та короткі концепти для стартапу у сфері штучного інтелекту для навчання."

temperatures = [0.0, 0.5, 1.0]

print("6.1 ДОСЛІДЖЕННЯ ПАРАМЕТРА TEMPERATURE")
for temp in temperatures:
    print(f"\nTemperature = {temp}")
    interaction_temp = client.interactions.create(
        model=MODEL,
        input=prompt_creative,
        generation_config={
            "temperature": temp
        }
    )
    print(interaction_temp.output_text.strip())

top_ps = [0.5, 0.8, 1.0]

print("\n6.2 ДОСЛІДЖЕННЯ ПАРАМЕТРА TOP_P")
for p in top_ps:
    print(f"\nTop_P = {p}")
    interaction_p = client.interactions.create(
        model=MODEL,
        input=prompt_creative,
        generation_config={
            "top_p": p
        }
    )
    print(interaction_p.output_text.strip())