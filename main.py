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

def get_product_names() -> List[str]:
    return [prod.get("nombre_producto", "") for prod in product_db if prod.get("nombre_producto")]

def find_similar_products(query: str) -> List[str]:
    similares = []
    for prod in product_db:
        nombre = prod.get("nombre_producto", "")
        if nombre and query.lower() in nombre.lower():
            similares.append(nombre)
    return similares

def is_relevant_query(query: str) -> bool:
    irrelevantes = [
        "clima", "política", "juegos", "presidente", "celebridad",
        "salud", "religión", "noticias", "deportes", "famosos"
    ]
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
                "temperature": 0.6
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
        return {"reply": "Solo puedo ayudarte con productos de nuestra tienda de zapatos. ¿Quieres ver opciones?"}

    if any(p in lower_query for p in ["hola", "buenas", "hey"]):
        return {"reply": "¡Hola! Soy Coco 🐾, tu asistente virtual. ¿Buscas algo en especial hoy?"}
    if any(p in lower_query for p in ["gracias", "muchas gracias"]):
        return {"reply": "¡Con gusto! Si necesitas más ayuda, estaré aquí 😊."}
    if "talla" in lower_query:
        tallas = get_shoe_sizes()
        return {
            "reply": f"Estas son las tallas disponibles actualmente: {', '.join(tallas)}."
            if tallas else "Por ahora no hay tallas disponibles."
        }

    productos_texto = []
    for prod in product_db:
        nombre = prod.get("nombre_producto", "Producto")
        tipo = prod.get("tipo_producto", "Desconocido")
        precio = prod.get("precio_producto", "N/A")
        tallas = [v.get("talla") for v in prod.get("variantes", []) if v.get("talla")]
        tallas_str = ", ".join(sorted(tallas)) if tallas else "No disponibles"
        productos_texto.append(f"- {nombre} ({tipo}) | Precio: {precio} | Tallas: {tallas_str}")

    productos_disponibles = get_product_names()
    similares = find_similar_products(query)

    extra_msg = ""
    if not similares and any(n in lower_query for n in productos_disponibles):
        extra_msg = "Lamentablemente no tenemos ese producto, pero puedo recomendarte otros similares."

    messages = [
        {
            "role": "system",
            "content": (
                "Eres Coco 🐾, un asistente de una tienda de zapatos online. Responde de forma amable, clara y sólo en base a la información proporcionada."
                " Si el producto no existe, ofrece sugerencias similares. No inventes ni hables de temas ajenos."
            )
        },
        {
            "role": "user",
            "content": "Estos son los productos disponibles:\n" + "\n".join(productos_texto)
        },
        {
            "role": "user",
            "content": (extra_msg + "\n" if extra_msg else "") + query
        }
    ]

    reply = call_llama3(messages)
    return {"reply": reply}
