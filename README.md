# Para testar esse projeto em sua máquina
### Esse projeto já foi testado com python versões: 3.11 e 3.13, no 3.14 ele não roda por conta da versão do chroma que não foi compatível

## Baixe e instale de acordo com seu sistema (Windows, Linux ou Mac).

Verifique se está funcionando:

* ollama --version

## Se for, por exemplo, llama3, rode:

* ollama pull llama3

## Clonar o repositório do projeto (Direto no vscode)

https://github.com/profcintiapinho/iacomllmrag.git

## Criar e ativar um ambiente virtual Python no terminal(prompt), não esqueça que precisar estar na pasta onde fez o clone

* python -m venv venv

* .\venv\Scripts\activate  



## Instalar as dependências

* pip install -r requirements.txt

<img width="639" height="445" alt="image" src="https://github.com/user-attachments/assets/61e153f8-72d6-4a96-a28e-5f0567292d4c" />

## Configurar o banco de dados / embeddings

* python setup_database.py

Sempre que você trocar os documentos de consulta, precisa executar esses códigos novamente:

<img width="962" height="520" alt="image" src="https://github.com/user-attachments/assets/b6631cee-8531-4c28-a820-6be9287bae12" />

## Entenda a estrutura de pastas

models/ → cuidam da parte de dados, por exemplo, como carregar documentos e transformar em embeddings.

controllers/ → fazem a “ponte” entre os dados (models) e a apresentação (views), definindo a lógica de uso do RAG (retrieval-augmented generation).

views/ → cuidam da interface com quem usa (seja uma API Flask, um retorno em terminal, etc.).

db/ → guarda o banco vetorial (ChromaDB) que armazena os embeddings e documentos processados.

documents/ → são os arquivos originais que você consulta.

run.py → ponto de entrada, é quem inicia tudo.

setup_database.py → prepara os dados antes de rodar o sistema.

## como rodar a aplicação?

* Depois de gerar os chunks e a pasta db é criada, agora é só rodar o projeto, digite no propt:
* python run.py

<img width="973" height="424" alt="image" src="https://github.com/user-attachments/assets/938cd1a4-3ad4-40ce-ae7a-7a9aa4f1c5ea" />

Em seguida abra o navegador com a api:
coloque o endereço no navegador e clique em "try it out"
E então comece a testar com os documentos que que possui, procure fazer perguntas que só estão contidas no documento, para garantir que estão sendo procuradas.

<img width="1500" height="432" alt="image" src="https://github.com/user-attachments/assets/7ebc7470-4f6d-44db-949e-9d8b6c0cad65" />


<img width="1074" height="601" alt="image" src="https://github.com/user-attachments/assets/9ba5a091-8a86-4b55-8957-70ea18fac543" />

Resposta
<img width="1456" height="555" alt="image" src="https://github.com/user-attachments/assets/1ff0a609-21f7-431e-ab8f-25fffd8cf77f" />


# Agora incluindo o Front:

Após rodar o python run.py, após rodar mudou um pouco a forma de abrir:

* http://127.0.0.1:5000/swagger - para abrir a api
* http://127.0.0.1:5000/frontend - para abrir o front




