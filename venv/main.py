
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")  # asegúrate de tener tu clave de HF en .env

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    inputs: List[Message]

@app.post("/chat")
def chat(request: ChatRequest):
    print(f"Received request: {request}")

    instruction = (
        "Responde únicamente al mensaje que te escriba el usuario. "
        "No simules una conversación ni devuelvas las preguntas del usuario. "
        "No generes respuestas adicionales ni diálogos completos y tampoco agregues signos que al usuario se le olvide colocar, como los signos de interrogación (¿?) o las comas (,) o los puntos (.) . "
        "No uses etiquetas como 'Cliente:' o 'Asistente:'. "
        "Responde solo en español y de forma breve y clara. "
        "Eres un asistente de una tienda de zapatos online. "
        "Los prompt que recibas no los autocompletes. "
        "Responde con un máximo de 2 líneas. Si la pregunta no está relacionada con calzado, responde: "
        "'Lo siento, no tengo información sobre eso.'\n\n"
    )

    # ✅ Aquí usas toda la conversación, no solo el último mensaje
    conversation_history = ""
    for msg in request.inputs:
        conversation_history += msg.content.strip() + "\n"

    full_prompt = instruction + conversation_history.strip()

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "inputs": full_prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 100,
            "do_sample": True,
            "top_p": 0.95,
        }
    }

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            result = response.json()
            generated_text = result[0]["generated_text"].replace(full_prompt, "").strip()
            return {"reply": generated_text}
        else:
            return {"error": f"Error Hugging Face {response.status_code}: {response.text}"}

    except requests.exceptions.RequestException as e:
        return {"error": f"Error en la conexión: {str(e)}"}
