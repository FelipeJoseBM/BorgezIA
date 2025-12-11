# app/__init__.py
from flask import Flask
from app.controllers.rag_controller import api as rag_namespace
from flask_restx import Api
from app.config import get_embedding_function
from app.models.rag_model import RAGModel

app = Flask(__name__)
api = Api(app, version='1.0', title='API RAG')

# Carrega RAGModel apenas uma vez
embedding_function = get_embedding_function()
rag_model_global = RAGModel(db_path="db", embedding_function=embedding_function)

api.add_namespace(rag_namespace, path='/rag')