from flask import Flask, send_from_directory
from app.controllers.rag_controller import api as rag_namespace
from flask_restx import Api
from app.config import get_embedding_function
from app.models.rag_model import RAGModel
from flask_cors import CORS  # <-- ADICIONAR

# Inicializa Flask
app = Flask(__name__, static_folder="app/views", template_folder="app/views")

# Habilitar CORS para o React
CORS(app, resources={r"*": {"origins": "http://localhost:5173"}})

# Inicializa o Swagger em /swagger
api = Api(app, version='1.0', title='API RAG', doc='/swagger')

# Registra o namespace da API
api.add_namespace(rag_namespace, path='/rag')

# Inicializa o RAG global uma vez
embedding_function = get_embedding_function()
rag_model_global = RAGModel(db_path="db", embedding_function=embedding_function)

# Rota para servir frontend
@app.route("/frontend")
def frontend():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(debug=True)
