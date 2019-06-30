import numpy as np
import tensorflow as tf
import re
import os

from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk

class QA():
  def __init__(self, contexto_previo, contexto_posterior, intencion, preguntas, respuestas):
    self.contexto_previo = contexto_previo
    self.contexto_posterior = contexto_posterior
    self.intencion = intencion # Obligatorio
    self.preguntas = preguntas
    self.respuestas = respuestas

class Chatbot():
  def __init__(self):
    self.stemmer = SnowballStemmer('spanish')
    self.stopwords = set([self.stemmer.stem(word) for word in stopwords.words('spanish')])
    self.stopwords.remove('si')
    self.stopwords.remove('no')
    self.stopwords.remove('eres')

    self.tam_secuencia = 10

    self.data = [
      QA('', 
        '',
        '',
        [''],
        ['No entendí tu consulta.']),
      QA('', 
        '', 
        'saludo',
        ['hola', 'hey como estas', 'hay alguien ahí'], 
        ['¿Hola cómo estas?', 'Dime en qué puedo ayudarte.', '¿En qué te puedo ayudar?']),
      QA('', 
        '',
        'descanso',
        ['que es descanso'], 
        ['Es una parte fundamental del entrenamiento y acondicionamiento físico, durante el organismo repone el desgaste sufrido en el ejercicio.']
      ),
      QA('',
        '',
        'importancia',
        ['por que debo dormir'], 
        ['Es muy importante para tu salud, o sino podrías tener algunos problemas de salud.']),
      QA('', 
        'presentacion',
        'presentacion',
        ['que eres', 'que haces', 'cual es tu funcion'],
        ['Soy un chatbot programado para ayudarte.']),
          QA('presentacion',
            '',
            'ayuda',
            ['ayudarme en que'],
            ['Te ayudo a identificar problemas de descanso.']),
      QA('', 
        '', 
        'agradecimiento',
        ['gracias', 'muchas gracias', 'gracias por tu ayuda', 'eres el mejor', 'me ayudaste mucho', 'gracias por la informacion'],
        ['Feliz de poder ayudarte :).']),
      QA('', 
        '', 
        'despedida',
        ['chau y gracias', 'nos vemos luego', 'hasta pronto', 'adios', 'hasta luego'], 
        ['Hasta pronto!', 'Nos vemos luego.', 'Espero haberte sido util.']),
      QA('', 
        '',
        'sueño',
        ['tengo problemas de sueño', 'no duermo bien', 'por que debo dormir'],
        ['Es muy importante para tu salud, puedes tener algunos problemas de salud.']
        ),
      QA('', 
        '',
        'enfermedad',
        ['que enfermedades se originan por falta de sueño', 'quiero saber si tengo alguna enfermedad'],
        ['Diabetes, hipertensión arterial, obesidad, depresión, ansiedad, entre muchos otros ...']),
      QA('cuestionario', 
        'cuestionario',
        'afirmacion',
        ['si', 'claro', 'es verdad', 'afirmativo', 'siempre', 'acertaste'],
        []),
      QA('cuestionario', 
        'cuestionario',
        'negacion',
        ['no', 'claro que no', 'es mentira', 'negativo', 'nunca', 'fallaste'],
        []),
      QA('', 
        'diabetes',
        'diabetes',
        ['quiero saber si tengo diabetes', 'como se si tengo diabetes', 'tendre diabetes? ayudame'],
        ['Empezemos por un cuestionario, ¿Te parece bien?']),
          QA('diabetes', 
            'cuestionario',
            'cuestionario',
            ['si', 'claro', 'por supuesto', 'ok', 'como quieras', 'prosigue', 'afirmativo', 'como gustes', ' seria lo mejor', 'vamos', 'gracias'],
            []),
          QA('diabetes', 
            'no_cuestionario',
            'no_cuestionario',
            ['no', 'claro que no', 'por supuesto que no', 'no quiero', 'detente', 'negativo', 'no me gusta', 'no seria lo mejor', 'no gracias'],
            ['Como gustes, puedes realizarlo cuando quieras.'])
    ]

    self.cuestionario = [
      '¿Has notado que últimamente te da más sed?',
      '¿Sientes más apetito que antes?',
      '¿Tienes fatiga?',
      '¿Tienes visión borrosa?',
      '¿Te han salido llagas que tardan en sanar?',
      '¿Tienes sobrepeso?',
      '¿Duermes de manera inadecuada? (Normal: 7-8 horas)',
      '¿Realizas poca actividad física?',
      '¿Tienes familiares que tuvieron diabetes?',
      '¿Tienes zonas de piel oscurecidas? (axilas y cuello)'
    ]

    for qa in self.data:
      qa.preguntas = [self.limpiarTexto(pregunta) for pregunta in qa.preguntas]

    self.idx_a_intencion = sorted(set([qa.intencion for qa in self.data]))
    self.intencion_a_idx = {t:i for i, t in enumerate(self.idx_a_intencion)}
    self.preguntas = [pregunta for qa in self.data for pregunta in qa.preguntas]
    self.idx_a_palabra = sorted(set([palabra for pregunta in self.preguntas for palabra in pregunta.split()]))
    self.idx_a_palabra.append('')
    self.palabra_a_idx = {p:i for i, p in enumerate(self.idx_a_palabra)}
    self.load_model()


  def load_model(self): 
    self.model = self.build_model(tam_vocabulario=len(self.idx_a_palabra), tam_intenciones=len(self.idx_a_intencion), embedding_dim=128, batch_size=1, rnn_units=200)
    self.model.load_weights('weights.h5')
    self.model.build(tf.TensorShape([1, None]))

  def obtenerQA(self, intencion):
    for qa in self.data:
      if qa.intencion == intencion:
        return qa
    return None

  def responder(self,contexto_previo, pregunta):
    pregunta = self.limpiarTexto(pregunta)
    x = [self.palabra_a_idx[palabra] for palabra in pregunta.split() if palabra in self.idx_a_palabra]
    x = np.pad(x, (0, self.tam_secuencia - len(x)), 'constant')
  
    predictions = self.model(tf.expand_dims(x, 0))
  
    max_val = 0 # criterio
    best_qa = self.data[0]
  
    for i, p in enumerate(predictions.numpy()[0]):
      qa = self.obtenerQA(self.idx_a_intencion[i])
    
      if qa.contexto_previo == contexto_previo or qa.intencion == '':
        if p>max_val:
          max_val = p
          best_qa = qa
    return best_qa

  def build_model(self,tam_vocabulario, tam_intenciones, embedding_dim, batch_size, rnn_units):
    model = tf.keras.Sequential([
      tf.keras.layers.Embedding(tam_vocabulario, embedding_dim, batch_input_shape=[batch_size, None]),
      tf.keras.layers.LSTM(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform'),
      tf.keras.layers.GRU(rnn_units, return_sequences=False, recurrent_initializer='glorot_uniform'),
      tf.keras.layers.Dense(tam_intenciones)
    ])
    return model

  def limpiarTexto(self,text):
    text = text.lower()                                                                # convertir texto a minúsculas
    text = re.sub(r'[^a-zá-ú0-9ñÑ\s]+',' ',text)                                        # obtener solo caracteres y numeros
    text = [self.stemmer.stem(w) for w in text.split() if self.stemmer.stem(w) not in self.stopwords] # remover tildes, estemizar, remover stopwords
    return ' '.join(text)

  def iniciar_conversacion(self, consulta = '', contexto_previo = '', cuestionario_index = 0,  cuestionario_activo = False, cuestionario_diabetes = 0):
    qa = self.responder(contexto_previo, consulta)
    contexto_previo = qa.contexto_posterior
    
    if cuestionario_activo == False and contexto_previo == 'cuestionario':
      cuestionario_index = 0
      cuestionario_activo = True
      cuestionario_diabetes = 0
      
    if cuestionario_activo == True:
      if qa.intencion == 'afirmacion':
        cuestionario_diabetes += 1
        cuestionario_index += 1
      elif qa.intencion == 'negacion':
        cuestionario_index += 1
      else:
        contexto_previo = 'cuestionario'
        
    if cuestionario_activo == True:
      if(cuestionario_index >= len(self.cuestionario)):
        cuestionario_activo = False
        contexto_previo = ''
        respuesta = '$fin_cuestionario'
      else:
        respuesta = self.cuestionario[cuestionario_index]
    else:
      respuesta = qa.respuestas

    return { 
      'consulta': consulta,
      'contexto_previo': contexto_previo,
      'cuestionario_index': cuestionario_index,
      'cuestionario_activo': cuestionario_activo,
      'cuestionario_diabetes': cuestionario_diabetes,
      'respuesta': respuesta
    }

chat = Chatbot()