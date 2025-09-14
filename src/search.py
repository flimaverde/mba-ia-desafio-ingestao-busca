import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt():
    try:        
        embeddings = GoogleGenerativeAIEmbeddings(
            task_type="RETRIEVAL_DOCUMENT", 
            model="gemini-embedding-001", 
            request_timeout=60
        )
        
        store = PGVector(
            embeddings=embeddings,
            collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
            connection=os.getenv("DATABASE_URL"),
            use_jsonb=True,
        )

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0
        )
        
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
        def query_chain(pergunta: str):
            
            results = store.similarity_search_with_score(pergunta, k=10)            
            
            contexto_docs = []
            for doc, score in results:
                contexto_docs.append(doc.page_content.strip())
            
            contexto = "\n\n".join(contexto_docs)
                        
            formatted_prompt = prompt.format_prompt(
                contexto=contexto,
                pergunta=pergunta
            )
                        
            response = llm.invoke(formatted_prompt.to_messages())
            
            return response.content

        return query_chain
        
    except Exception as e:
        print(f"Erro ao inicializar o sistema de busca: {e}")
        return None