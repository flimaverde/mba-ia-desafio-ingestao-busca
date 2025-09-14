from search import search_prompt
import sys

def main():          
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        print("\nVerifique se:")
        print("- O PostgreSQL está rodando (docker compose up -d)")
        print("- As variáveis de ambiente estão configuradas (.env)")
        print("- A ingestão foi executada (python src/ingest.py)")
        return
    
    print("Sistema inicializado com sucesso!")
    print("\nDigite 'sair', 'exit' ou 'quit' para encerrar.")
    print("="*60)
        
    while True:
        try:
     
            print("\nFaça sua pergunta:")
            pergunta = input("\nPERGUNTA: ").strip()
            
     
            if pergunta.lower() in ['sair', 'exit', 'quit', '']:
                print("\nEncerrando o chat")
                break
            
            print("\nBuscando informações...")
                 
            try:
                resposta = chain(pergunta)
                print(f"RESPOSTA: {resposta}")
            except Exception as e:
                print(f"Erro ao processar pergunta: {e}")
                print("Tente novamente com uma pergunta diferente.")
            
            print("\n" + "-"*60)
            
        except KeyboardInterrupt:
            print("\n\nChat interrompido pelo usuário.")
            break
        except EOFError:
            print("\n\nEncerrando o chat.")
            break
        except Exception as e:
            print(f"\nErro inesperado: {e}")            

if __name__ == "__main__":
    main()