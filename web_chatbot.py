import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

chatbot = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def respond(message, history):
    response = chatbot(message, max_new_tokens=100)[0]['generated_text']
    return response

gr.ChatInterface(fn=respond).launch()