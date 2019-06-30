from flask import request, jsonify,Flask
from utils import chat
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)


@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    content = request.json

    consulta = content['consulta']
    contexto_previo = content['contexto_previo']
    cuestionario_index = content['cuestionario_index']    
    cuestionario_activo = content['cuestionario_activo']
    cuestionario_diabetes = content['cuestionario_diabetes']
    __json__ = chat.iniciar_conversacion(consulta=consulta, contexto_previo=contexto_previo, cuestionario_index=cuestionario_index, cuestionario_activo=cuestionario_activo, cuestionario_diabetes=cuestionario_diabetes)

    return jsonify(__json__)

if __name__ == "__main__":
    app.run(debug=True)