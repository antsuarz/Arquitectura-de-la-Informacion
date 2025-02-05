import re
import json
import torch 
from transformers import BertTokenizer, BertModel
from llama_index.llms.ollama import Ollama

local_llm = Ollama(
    model="llama3.2",
    request_timeout=600.0,
    temperature=0.1,
    system_prompt='responde siempre en español'
)

tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')
model = BertModel.from_pretrained('bert-large-cased-whole-word-masking')

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
        pkeys = obtener_palabras_clave(parrafo)
        query = f"Dame la respuesta en un JSON respuesta:palabra_clave. Es importante que cumplas con el formato de respuesta ya que lo tengo que procesar: Te voy a dar tres palabras clave: {pkeys[0]}, {pkeys[1]}, {pkeys[2]}. Por favor, dame como respuesta unicamente la que consideres que puede servir para titular este parrafo,  \n\n{parrafo}."
        result = local_llm.complete(query)
        pkey = json.loads(str(result)).get('palabra_clave', 0)
        with open("./salidas/" + nombre, 'a', encoding='utf-8') as file:
            file.write(f"<h2>{pkey}</h2>\n")
            file.write(f"<p>{parrafo}</p>\n")
    print("HTML GENERADO")


def obtener_palabras_clave(oracion): 
    oracion = re.sub(r'[^\w\s]', '', oracion)
    ids = tokenizer(oracion, return_tensors='pt', add_special_tokens=True)
    with torch.no_grad():
        bert_output = model(**ids)
    last_hidden_state = bert_output.last_hidden_state
    tokens = tokenizer.convert_ids_to_tokens(ids['input_ids'][0])
    word_importance = last_hidden_state[0].norm(dim=1).cpu().numpy()
    word_score_pairs = list(zip(tokens, word_importance))
    word_score_pairs = [pair for pair in word_score_pairs if pair[0] not in tokenizer.all_special_tokens]
    word_score_pairs = sorted(word_score_pairs, key=lambda x: x[1], reverse=False)
    palabras_clave = [pair[0] for pair in word_score_pairs[:3]]
    
    return palabras_clave


procesar_html("6.html")