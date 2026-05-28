from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model_path = "./saved_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path)

# -----------------------------------
# CHAT LOOP
# -----------------------------------

print("Finance AI Assistant Started")
print("Type 'exit' to quit\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    prompt = f"Question: {question}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
    **inputs,
    max_new_tokens=40,
    temperature=0.1,
    do_sample=False,
    repetition_penalty=1.2,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # REMOVE PROMPT FROM OUTPUT
    answer = response.replace(prompt, "").strip()

    print("\nAI:", answer)
    print("\n" + "-" * 50)