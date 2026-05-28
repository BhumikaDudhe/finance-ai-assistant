from flask import Flask, render_template, request, jsonify

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import torch

# -----------------------------------
# CREATE FLASK APP
# -----------------------------------

app = Flask(__name__)

# -----------------------------------
# BASE MODEL
# -----------------------------------

base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# -----------------------------------
# LOAD TOKENIZER
# -----------------------------------

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# -----------------------------------
# LOAD BASE MODEL
# -----------------------------------

print("Loading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float32
)

# -----------------------------------
# LOAD FINE-TUNED MODEL
# -----------------------------------

print("Loading fine-tuned adapter...")

model = PeftModel.from_pretrained(
    base_model,
    "./saved_model"
)

print("Finance AI Model Loaded Successfully!")

# -----------------------------------
# HOME PAGE
# -----------------------------------

@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------------
# CHAT API
# -----------------------------------

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    user_message = data["message"]

    # PROMPT

    prompt = f"Question: {user_message}\nAnswer:"

    # TOKENIZE

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    # GENERATE RESPONSE

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        temperature=0.1,
        do_sample=False,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )

    # DECODE

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # CLEAN RESPONSE

    answer = response.replace(prompt, "").strip()

    if "Question:" in answer:
        answer = answer.split("Question:")[0]

    if "Answer:" in answer:
        answer = answer.split("Answer:")[0]

    answer = answer.strip()

    # RETURN JSON

    return jsonify({
        "response": answer
    })

# -----------------------------------
# RUN SERVER
# -----------------------------------

if __name__ == "__main__":

    app.run(
        debug=False
    )