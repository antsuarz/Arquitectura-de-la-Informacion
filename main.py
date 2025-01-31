import re
import json
from llama_index.llms.ollama import Ollama
 
local_llm = Ollama(
    model="llama3.2",
    request_timeout=600.0, 
    temperature=0.1,
    system_prompt='responde siempre en español') 

def obtener_oraciones(texto):
    oraciones = texto.split(".")
    return oraciones

def procesar_html(nombre):
    with open("./muestra/"+ nombre, 'r', encoding='utf-8') as file:
        html_content = file.read()
    texto = re.findall(r'<p>(.*?)</p>', html_content, re.DOTALL)
    #obtener_oraciones(str(texto))
    query = f"Te voy a introducir un texto, dividemelo en parrafos segun el tema del que estén hablando (varias oraciones pueden estar hablando de lo mismo). Es muy importante que me lo devuelvas en un unico JSON con clave numero de parrafo (por ejemplo: 1). Por favor, devuelveme solo un JSON, nada más. Asegurate de que tu respuesta empieza y acaba por una llave, ya que tengo que procesarlo despues:\n\n{texto}"
    result = local_llm.complete(query)
    parsed_response = json.loads(str(result))
    for key, value in parsed_response.items():
        print(f"Párrafo {key}: {value}\n") 
        with open("./salidas/" + nombre, 'a', encoding='utf-8') as file:
            file.write(f"<p>{value}</p>\n")

procesar_html("1.html")