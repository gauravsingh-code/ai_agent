from llm.gemini import GeminiClient
from agent.agent import Agent

def main():

    llm = GeminiClient()
    # response = llm.generate("explain what an agent in simple terms.")
    
    agent = Agent(llm)
    print("AI Agent")
    print("type 'exit' to quit.\n")

    

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = agent.chat(user_input)
        print(f"\nAI Agent: {response}")


if __name__ == "__main__":
    main()


    
