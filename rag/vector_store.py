import chromadb


class VectorStore:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        self.collection = self.client.get_or_create_collection(
            name="documents"
        )

    def add(
        self,
        documents: list[str],
        embeddings,
        ids: list[str]
    ):
        self.collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            ids=ids
        )

    def search(
        self,
        embedding,
        top_k: int = 5
    ):
        return self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k
        )