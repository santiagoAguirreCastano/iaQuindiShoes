import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
product_db: List[Any] = []


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    inputs: List[Message]


class ProductRequest(BaseModel):
    products: List[Any]


# Helpers
def get_shoe_sizes() -> List[str]:
    sizes = set()
    for prod in product_db:
        for var in prod.get("variantes", []):
            talla = var.get("talla")
            if talla:
                sizes.add(talla)
    return sorted(list(sizes))


def is_relevant_query(query: str) -> bool:
    irrelevantes = ["clima", "política", "juegos", "presidente", "celebridad", "salud", "religión"]
    return not any(pal in query.lower() for pal in irrelevantes)


def call_llama3(messages: List[dict]) -> str:
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama3-70b-8192",
                "messages": messages,
                "temperature": 0.7
            }
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        return "Lo siento, hubo un problema al responder. Intenta nuevamente."
    except Exception as e:
        return f"Error al conectar con la IA: {str(e)}"


@app.post("/process-products")
def process_products(request: ProductRequest):
    global product_db
    product_db = request.products

    sizes_summary = []
    for product in product_db:
        nombre = product.get("nombre_producto", "Producto")
        tallas = [v.get("talla") for v in product.get("variantes", []) if v.get("talla")]
        if tallas:
            sizes_summary.append(f"{nombre}: {', '.join(sorted(tallas))}")

    if not sizes_summary:
        return {"reply": "No se encontraron tallas disponibles."}

    return {
        "reply": "Tallas disponibles por producto:\n" + "\n".join(sizes_summary)
    }


@app.post("/chat")
def chat(request: ChatRequest):
    if not request.inputs:
        return {"reply": "Envía un mensaje válido para ayudarte."}

    query = request.inputs[-1].content.strip()
    lower_query = query.lower()

    if not is_relevant_query(query):
        return {"reply": "Solo puedo responder sobre nuestros productos de calzado. ¿En qué más te ayudo?"}

    # Respuestas rápidas
    if any(p in lower_query for p in ["hola", "buenas", "hey"]):
        return {"reply": "¡Hola! ¿En qué puedo ayudarte con nuestros zapatos?"}
    if any(p in lower_query for p in ["gracias", "muchas gracias"]):
        return {"reply": "¡Con gusto! Si necesitas más ayuda, estaré aquí."}
    if "talla" in lower_query:
        tallas = get_shoe_sizes()
        return {
            "reply": f"Tallas disponibles: {', '.join(tallas)}." if tallas else "No hay tallas disponibles."
        }

    # Construir contexto de productos
    productos_texto = []
    for prod in product_db:
        nombre = prod.get("nombre_producto", "Producto")
        tipo = prod.get("tipo_producto", "Desconocido")
        precio = prod.get("precio_producto", "N/A")
        tallas = [v.get("talla") for v in prod.get("variantes", []) if v.get("talla")]
        tallas_str = ", ".join(sorted(tallas)) if tallas else "No disponibles"
        productos_texto.append(f"- {nombre} ({tipo}) | Precio: {precio} | Tallas: {tallas_str}")

    messages = [
        {"role": "system", "content": "Eres un asistente amigable y profesional de una tienda de zapatos online. Responde con claridad y sin inventar."},
        {"role": "user", "content": "Estos son los productos disponibles:\n" + "\n".join(productos_texto)},
        {"role": "user", "content": query}
    ]

    reply = call_llama3(messages)
    return {"reply": reply}
