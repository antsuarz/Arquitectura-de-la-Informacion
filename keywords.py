from transformers import BertTokenizer, BertModel
import torch
import re
 
tokenizer = BertTokenizer.from_pretrained('bert-large-cased-whole-word-masking')
model = BertModel.from_pretrained('bert-large-cased-whole-word-masking')

def obtener_palabras_clave(oracion): 
    oracion = re.sub(r'[^\w\s]', '', oracion)
    inputs = tokenizer(oracion, return_tensors='pt', add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
     
    last_hidden_state = outputs.last_hidden_state
     
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
     
    word_importance = last_hidden_state[0].norm(dim=1).cpu().numpy()
     
    word_score_pairs = list(zip(tokens, word_importance))
     
    word_score_pairs = [pair for pair in word_score_pairs if pair[0] not in tokenizer.all_special_tokens]
     
    word_score_pairs = sorted(word_score_pairs, key=lambda x: x[1], reverse=False)
 
    palabras_clave = [pair[0] for pair in word_score_pairs[:5]]
    
    return palabras_clave


oracion = "Michael Schumacher, considerado uno de los mejores pilotos de la historia, fue el primer campeón del siglo XXI.Después de su último título con Ferrari en 2000, defendió el campeonato en 2001, y en 2002 y 2004 dominócompletamente la F1, consolidando su legado.Posteriormente, Fernando Alonso, con Renault, rompió el dominio de Schumacher al ganar los campeonatos en 2005 y 2006, convirtiéndose en el primer campeón español de la historia de la F1."
palabras_clave = obtener_palabras_clave(oracion)
print("Palabras clave:", palabras_clave)
