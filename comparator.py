import re
import json
from llama_index.llms.ollama import Ollama

local_llm = Ollama(
    model="llama3.2",
    request_timeout=600.0,
    temperature=0.1,
    system_prompt='responde siempre en español'
)
 
def obtener_oraciones(texto): 
    texto_limpio = re.sub(r'\s+', ' ', texto).strip()
    oraciones = re.split(r'(?<=\.)\s*', texto_limpio)  
    return [oracion for oracion in oraciones if oracion]
 
def calcular_similitud(oracion1, oracion2): 
    query = f"Te voy a dar dos oraciones. Por favor, dame como respuesta unicamente una valoración de su similitud, con un numero entre 0 y 1 empleando BERT. Dame la respuesta en un JSON respuesta:numero. Es importante que cumplas con el formato de respuesta ya que lo tengo que procesar\n\n{oracion1}\n{oracion2}."
    result = local_llm.complete(query)
    try:
        similitud = json.loads(str(result)).get('similitud', 0)
    except json.JSONDecodeError:
        similitud = 0    
    return similitud

def obtener_parrafos(oraciones):
    parrafos = []
    for i in range(len(oraciones) - 1):
        if (i == 0):
            parrafos.append(oraciones[i])
        print(oraciones[i])
        print(oraciones[i+1])
        similitud = calcular_similitud(oraciones[i], oraciones[i+1])
        print(f"Similitud entre Oración {i} y Oración {i+1}: {similitud}\n")
        if(similitud > 0.8):
            parrafos[-1] = parrafos[-1] + oraciones[i+1]
        else:
            parrafos.append(oraciones[i+1])
    return parrafos

def procesar_html(nombre):
    with open(f"./muestra/{nombre}", 'r', encoding='utf-8') as file:
        html_content = file.read()    
    texto = re.findall(r'<p>(.*?)</p>', html_content, re.DOTALL)
    parrafos = obtener_parrafos(obtener_oraciones(str(texto[0])))

    for parrafo in parrafos:
        with open("./salidas/" + nombre, 'a', encoding='utf-8') as file:
            file.write(f"<p>{parrafo}</p>\n")
    
procesar_html("1.html")
