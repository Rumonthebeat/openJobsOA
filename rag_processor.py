import pandas as pd
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI

from config import ChatGPTConfig


class RAGProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(api_key=ChatGPTConfig.CHAT_GPT_KEY)
        self.gpt_client = OpenAI(api_key=ChatGPTConfig.CHAT_GPT_KEY)
        self.index = None
        self.metadata = []

    # init with loading csv file to faiss
    def initialize_system(self, csv_path):

        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise ValueError(f"CSV文件不存在: {csv_path}")

        # split to chucks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        documents = []
        for _, row in df.iterrows():
            chunks = text_splitter.split_text(row["job_description"])
            for chunk in chunks:
                documents.append({
                    "text": chunk,
                    "job_id": row["job_id"],
                    "source": "job_description"
                })

        # embedded vector
        texts = [doc["text"] for doc in documents]
        embeddings = self.embeddings.embed_documents(texts)

        # create index
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

        # save metadata
        self.metadata = documents

    # search by similarity
    def search(self, query, k):
        if not query or k <= 0:
            raise ValueError("Query is empty or k less than 0.")

        query_embedding = self.embeddings.embed_query(query)

        # search in faiss
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'),
            k
        )
        return [self.metadata[i] for i in indices[0]]

    # generate response
    def generate_response(self, query, context):
        context_str = "\n\n".join(
            [f"[SourceID:{doc['job_id']}]\n{doc['text']}"
             for doc in context]
        )

        try:
            response = self.gpt_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """
                            You are a professional job information assistant:
                            Rules:
                            1. Extract information ONLY from the provided context.
                            2. The source ID of the information must be labeled (e.g., [SourceID:123]).
                            3. Keep your answer concise and accurate. Do not add any external information or assumptions.
                            """},
                    {"role": "user", "content": f"Question：{query}\nContext：{context_str}"}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Fail to generate response: {str(e)}")