# setup_database.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shutil
import csv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader  # CSVLoader não é usado, pode remover
from langchain_text_splitters import CharacterTextSplitter
from langchain.schema import Document

from app.config import get_embedding_function, MODEL_NAME

# Pasta dos documentos
DOC_DIR = "documents"

# Caminho do banco de dados vetorial
DB_DIR = "db"

# Remove o banco anterior para começar do zero
# if os.path.exists(DB_DIR):
#     shutil.rmtree(DB_DIR)

documents = []
for file_name in os.listdir(DOC_DIR):
    file_path = os.path.join(DOC_DIR, file_name)

    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file_name.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())
    elif file_name.endswith(".csv"):
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["Problema"]
                metadata = {"URL": row.get("URL", "")}
                documents.append(Document(page_content=text, metadata=metadata))
    else:
        print(f"Atenção: arquivo ignorado (não suportado): {file_name}")

# Divide os documentos em chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

print(f"Total de documentos carregados: {len(documents)}")
print(f"Total de chunks criados: {len(docs)}")

# Cria os embeddings usando o config centralizado
print(f"Criando embeddings com o modelo '{MODEL_NAME}'...")
embedding_function = get_embedding_function()

# Cria o banco vetorial
print("Armazenando no banco de dados vetorial...")
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_function,
    persist_directory=DB_DIR
)
print("Banco de dados vetorial criado e armazenado com sucesso.")