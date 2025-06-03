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
            # Convertir a string y agregar el símbolo €
            prices.append(f"{precio} €")
    return prices

@app.post("/process-products")
def process_products(request: ProductRequest):
    global product_db
    # Almacena los productos reales recibidos en la variable global
    product_db = request.products
    print("Payload recibido:", request.dict())

    product_instruction = (
        "Eres un asistente de una tienda de zapatos online. "
        "Analiza la siguiente información de productos y genera una respuesta breve y clara sobre ellos:\n\n"
    )
    
    # Construir los detalles del producto con datos reales
    product_details = ""
    for product in request.products:
        # Puedes ajustar qué campos incluir en la respuesta
        product_details += f"ID: {product.get('id_producto')}, Tipo: {product.get('tipo_producto')}, Nombre: {product.get('nombre_producto')}, Precio: {product.get('precio_producto')} €, "
        variantes = product.get("variantes", [])
        if variantes:
            tallas = ", ".join([var.get("talla", "") for var in variantes])
            product_details += f"Tallas: {tallas}"
        product_details += "\n"
    
    full_prompt = product_instruction + product_details.strip()

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

@app.post("/chat")
def chat(request: ChatRequest):
    try:
        print(f"Received request: {request}")
    except Exception as e:
        return {"reply": "Error al procesar la solicitud."}

    # Validar que existan mensajes y que cada uno tenga contenido
    if not request.inputs or not all(hasattr(msg, "content") and msg.content for msg in request.inputs):
         return {"reply": "La solicitud debe incluir mensajes válidos."}

    # Reconstruir la consulta completa y convertir a minúsculas para validar
    conversation_history = " ".join([msg.content.strip() for msg in request.inputs])
    lower_conversation = conversation_history.lower()

    # Lista de palabras clave generales relacionadas con calzado
    keywords = ["zapato", "calzado", "bota", "sandalia", "zapatilla","tallas","precio", "nombre", "tipos de zapatos", "disponibles"]
    if not any(keyword in lower_conversation for keyword in keywords):
         return {"reply": "Lo siento, no tengo información sobre eso."}

    # Instrucción base para el asistente:
    # - Responder únicamente con la información de productos almacenada en la tienda (product_db).
    # - No autogenerar preguntas adicionales ni repetir la consulta original.
    base_instruction = (
        "Eres un asistente de una tienda de zapatos online. "
        "Debes responder únicamente basándote en la información disponible en la tienda, que se obtiene de la base de datos real. "
        "Utiliza exclusivamente los siguientes datos: "
        "Tipos: {tipos}, Nombres: {nombres}, Tallas: {tallas}, Precios: {precios}. "
        "No incluyas preguntas adicionales ni repitas la consulta del usuario. "
        "Responde en forma breve y precisa. "
        "Si la consulta no se relaciona directamente con la información de la tienda, responde: 'Lo siento, no tengo información sobre eso.'\n\n"
    ).format(
         tipos=", ".join(get_shoe_types()),
         nombres=", ".join(get_shoe_names()),
         tallas=", ".join(get_shoe_sizes()),
         precios=", ".join(get_shoe_prices())
    )

    # Construir additional_info según palabras clave específicas:
    additional_info = ""
    # Si se pregunta específicamente por tipos
    if "tipos de zapatos" in lower_conversation or "disponibles" in lower_conversation:
         additional_info = f"Tipos disponibles: {', '.join(get_shoe_types())}.\n\n"
    # Si se pregunta por nombres
    if "nombre" in lower_conversation:
         additional_info = f"Nombres disponibles: {', '.join(get_shoe_names())}.\n\n"
    # Si se pregunta por tallas
    if "talla" in lower_conversation:
         additional_info = f"Tallas disponibles: {', '.join(get_shoe_sizes())}.\n\n"
    # Si se pregunta por precios
    if "precio" in lower_conversation or "valen" in lower_conversation:
         additional_info = f"Precios disponibles: {', '.join(get_shoe_prices())}.\n\n"

    full_prompt = base_instruction + additional_info + "Consulta: " + conversation_history

    headers = {
         "Authorization": f"Bearer {HF_API_KEY}",
         "Content-Type": "application/json"
    }

    data = {
         "inputs": full_prompt,
         "parameters": {
              "temperature": 0.5,
              "max_new_tokens": 100,
              "do_sample": False,
              "top_p": 0.9,
         }
    }

    try:
         response = requests.post(
             "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
             headers=headers,
             json=data
         )
         if response.status_code == 200:
             try:
                 result = response.json()
             except Exception as e:
                 return {"reply": "Error al interpretar la respuesta de la IA."}
             
             generated_text = result[0]["generated_text"].replace(full_prompt, "").strip()
             
             # Validaciones específicas para asegurar que la respuesta incluya lo solicitado:
             if ("precio" in lower_conversation or "valen" in lower_conversation) and "€" not in generated_text:
                 return {"reply": "La respuesta generada no contiene la información de precios de la tienda."}
             if ("talla" in lower_conversation) and not any(size in generated_text for size in get_shoe_sizes()):
                 return {"reply": "La respuesta generada no contiene la información de tallas de la tienda."}
             if ("nombre" in lower_conversation) and not any(name.lower() in generated_text.lower() for name in get_shoe_names()):
                 return {"reply": "La respuesta generada no contiene la información de nombres de la tienda."}
             if ("tipos de zapatos" in lower_conversation) and not any(tp.lower() in generated_text.lower() for tp in get_shoe_types()):
                 return {"reply": "La respuesta generada no contiene la información de tipos de zapatos de la tienda."}
             
             # Evitar respuestas que autogeneren preguntas adicionales
            
             
             return {"reply": generated_text}
         else:
             return {"error": f"Error Hugging Face {response.status_code}: {response.text}"}
    except requests.exceptions.RequestException as e:
         return {"error": f"Error en la conexión: {str(e)}"}
