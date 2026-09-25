def build_rag_prompt(question:str, contexts:list[str]) -> str:
    context = "\n\n---\n\n".join(contexts)
    return f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:
"I don't have enough information in the provided documents."

Do not invent or assume information.

Context:
{context}

User question:
{question}

Answer:
""" 

