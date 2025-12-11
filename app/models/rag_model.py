import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RAGModel:
    def __init__(self, db_path, embedding_function, max_chunks=5, max_context_chars=3000):
        self.vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embedding_function
        )
        self.retriever = self.vectorstore.as_retriever()
        self.llm = Ollama(model="llama3")
        self.max_chunks = max_chunks
        self.max_context_chars = max_context_chars

    def get_rag_response(self, question):
        # Recupera os chunks mais relevantes
        retrieved_docs = self.retriever.get_relevant_documents(question)[:self.max_chunks]

        # Monta o contexto concatenando chunks
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        if len(context) > self.max_context_chars:
            context = context[:self.max_context_chars] + "\n\n... [texto truncado]"

        # Prompt instrui o LLM a usar documentos primeiro, mas fallback se necessário
        template = """
        
    Você é um professor de investimentos para adolescentes.
    
    Seja breve e direto.
 
    Seja descontraído e use gírias.
    
    Responda a pergunta usando APENAS o contexto abaixo.

    Se a resposta não estiver no contexto, diga: "Não sei a resposta para isso."

   

    Contexto:
    {context}

    Pergunta:
    {question}

    Resposta:
    """

        prompt = ChatPromptTemplate.from_template(template)
        output_parser = StrOutputParser()
        chain = prompt | self.llm | output_parser

        # Executa a cadeia
        return chain.invoke({"context": context, "question": question})