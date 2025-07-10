import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

HF_API_KEY = os.getenv("HF_API_KEY")  # asegúrate de tener tu clave de HF en .env

# Variable global para almacenar los productos reales traídos de la BD
product_db: List[Any] = []

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    inputs: List[Message]

class ProductRequest(BaseModel):
    products: List[Any]

# Funciones helper que extraen información del producto real
def get_shoe_types() -> List[str]:
    global product_db
    types = set()
    for prod in product_db:
        tipo = prod.get("tipo_producto")
        if tipo:
            types.add(tipo.capitalize())
    return list(types)

def get_shoe_names() -> List[str]:
    global product_db
    names = []
    for prod in product_db:
        nombre = prod.get("nombre_producto")
        if nombre:
            names.append(nombre)
    return names

def get_shoe_sizes() -> List[str]:
    global product_db
    sizes = set()
    for prod in product_db:
        variantes = prod.get("variantes", [])
        for var in variantes:
            talla = var.get("talla")
            if talla:
                sizes.add(talla)
    return list(sizes)

def get_shoe_prices() -> List[str]:
    global product_db
    prices = []
    for prod in product_db:
        precio = prod.get("precio_producto")
        if precio:
            prices.append(f"{precio} $")
    return prices

# Función para filtrar las consultas no relacionadas con productos
def is_relevant_query(query: str) -> bool:
    # Definir temas que no tienen que ver con la tienda de zapatos
    irrelevant_keywords = [
        "clima", "fútbol", "noticias", "política", "presidente", "celebridad", "deportes",
        "película", "música", "chisme", "recomendaciones", "anime", "juegos", "economía", 
        "historia", "geografía", "viajes", "cultura", "religión", "salud", "educación", 
        "matemáticas", "ciencia", "filosofía", "programación", "computación", "astronomía", 
        "tecnología", "literatura", "arte", "personajes históricos", "famosos", "espectáculos", 
        "concursos", "polémica", "debate", "sociales", "polemica", "viajes", "celebridades"
    ]
    return not any(keyword in query.lower() for keyword in irrelevant_keywords)

@app.post("/process-products")
def process_products(request: ProductRequest):
    global product_db
    # Almacena los productos reales recibidos en la variable global
    product_db = request.products
    print("Payload recibido:", request.dict())

    # Instrucción modificada para obtener solo las tallas
    product_instruction = (
        "Eres un asistente de una tienda de zapatos online. "
        "Analiza la siguiente información de productos y responde SOLO con las tallas disponibles para cada uno, sin agregar detalles extra:\n\n"
        "Asegúrate de incluir SOLO las tallas disponibles para los productos y nada más."
    )

    # Construir los detalles de las tallas de los productos
    sizes_details = ""
    for product in request.products:
        variantes = product.get("variantes", [])
        tallas = ", ".join([var.get("talla", "") for var in variantes])
        if tallas:
            sizes_details += f"Tallas disponibles: {tallas}\n"

    # Solo devolvemos las tallas
    full_prompt = product_instruction + sizes_details.strip()

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
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
,
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            result = response.json()
            generated_text = result[0]["generated_text"].replace(full_prompt, "").strip()
            generated_text = generated_text.strip('"')  # elimina las comillas dobles al inicio y al final

            # Cortar la respuesta por salto de línea y usar solo la primera parte útil
            generated_text = generated_text.split("\n")[0].strip()

            # Si la respuesta no es útil o no tiene tallas, devuelve un mensaje estándar
            if not generated_text or "no tengo" in generated_text.lower():
                return {"reply": "No encontré información sobre las tallas disponibles. ¿Podrías intentar con otra pregunta?"}

            return {"reply": generated_text}
        else:
            return {"error": f"Error Hugging Face {response.status_code}: {response.text}"}

    except requests.exceptions.RequestException as e:
        return {"error": f"Error en la conexión: {str(e)}"}

@app.post("/chat")
def chat(request: ChatRequest):
    if not request.inputs or not all(hasattr(msg, "content") and msg.content for msg in request.inputs):
        return {"reply": "La solicitud debe incluir mensajes válidos."}

    user_query = request.inputs[-1].content.strip()
    lower_query = user_query.lower()

    # Filtrar preguntas irrelevantes
    if not is_relevant_query(user_query):
        return {"reply": "Lo siento, solo puedo responder preguntas relacionadas con nuestra tienda de zapatos. ¿En qué más te puedo ayudar?"}

    # Respuestas rápidas predefinidas
    saludos = ["hola", "buenas", "hey", "qué tal"]
    agradecimientos = ["gracias", "muchas gracias", "te lo agradezco"]

    if any(s in lower_query for s in saludos):
        return {"reply": "¡Hola! ¿En qué puedo ayudarte con nuestros zapatos?"}
    if any(a in lower_query for a in agradecimientos):
        return {"reply": "¡De nada! Si necesitas algo más, estaré encantado de ayudarte."}

    # Ajustamos para que solo se respondan las tallas disponibles sin agregar detalles extra
    if "tallas" in lower_query:
        available_sizes = get_shoe_sizes()
        if available_sizes:
            return {"reply": f"Las tallas disponibles son: {', '.join(available_sizes)}."}
        else:
            return {"reply": "No tenemos tallas disponibles en este momento."}

    # Datos de productos en formato estructurado por producto
    prompt_header = (
        "Eres un asistente para una tienda online de zapatos. Usa los siguientes productos para responder la consulta del cliente.\n\n"
    )

    product_details = ""
    for i, product in enumerate(product_db):
        nombre = product.get("nombre_producto", "Desconocido")
        tipo = product.get("tipo_producto", "Desconocido")
        precio = f"{product.get('precio_producto')} $" if product.get("precio_producto") else "Precio no disponible"
        tallas = ", ".join(var.get("talla", "") for var in product.get("variantes", [])) or "No disponibles"
        product_details += (
            f"Producto {i+1}:\n"
            f"- Tipo: {tipo}\n"
            f"- Nombre: {nombre}\n"
            f"- Tallas: {tallas}\n"
            f"- Precio: {precio}\n\n"
        )

    prompt = (
        prompt_header + product_details + f"Consulta del cliente: {user_query}\n"
        "Responde con la información relevante sin inventar ni repetir la pregunta."
    )

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.4,
            "max_new_tokens": 150,
            "do_sample": False,
            "top_p": 0.9,
        }
    }

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
,
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            return {"error": f"Error Hugging Face {response.status_code}: {response.text}"}

        result = response.json()
        generated_text = result[0]["generated_text"].replace(prompt, "").strip()
        generated_text = generated_text.strip('"')  # elimina las comillas dobles al inicio y al final

        # Cortar la respuesta por salto de línea y usar solo la primera parte útil
        generated_text = generated_text.split("\n")[0].strip()

        # ✅ Si la respuesta no es útil, se retorna un mensaje de error
        if not generated_text or "no tengo" in generated_text.lower():
            return {"reply": "No encontré información específica para esa consulta. ¿Podrías intentar con otra pregunta?"}

        return {"reply": generated_text}

    except requests.exceptions.RequestException as e:
        return {"error": f"Error en la conexión: {str(e)}"}
