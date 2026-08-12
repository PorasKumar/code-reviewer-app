from pinecone import Pinecone
from pinecone import ServerlessSpec
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

class Upsert:
    def __init__(self, chunks, dense_embeddings, sparse_embeddings, namespace):
        if (chunks and dense_embeddings and sparse_embeddings and namespace):
            self.chunks = chunks
            self.dense_embeddings = dense_embeddings
            self.sparse_embeddings = sparse_embeddings
            self.namespace = namespace
        else:
            raise ValueError(f"Error in Upserting.Valid Data i.e., Chunks, dense & sparse embeddings, namespace are required for upserting!")
        
        try:
            pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            self.index_name = os.getenv("PINECONE_INDEX_NAME")
            if not pc.has_index(self.index_name):
                pc.create_index(
                    name=self.index_name,
                    dimension=384,
                    spec= ServerlessSpec(cloud="aws",region="us-east-1"),
                    metric="dotproduct",
                    )
            self.index = pc.Index(self.index_name)
        
        except Exception as e:
            print(f"Error in initialising Pinecone database: \n\n{e}")
            raise RuntimeError(f"Error in initialising Pinecone database: \n{e}")

    def upsert_data(self):
        try:
            upsert_payload = []

            for i, (chunk, dense_emb, sparse_emb) in enumerate(zip(self.chunks,self.dense_embeddings,self.sparse_embeddings)):
                #metadata
                mtdata = chunk.metadata.copy() if hasattr(chunk, "metadata") else {} #if metadata exist else nothing
                mtdata["text"] = chunk.page_content
                mtdata["doc_index"]  = i
                mtdata["content_length"] = len(chunk.page_content)

                #payload below
                record = {
                    "id": f"doc_{uuid.uuid4().hex[0:8]}_{i}",
                    "values":dense_emb,
                    "sparse_values":sparse_emb,
                    "metadata": mtdata,
                }

                upsert_payload.append(record)

            #Upsert in batches of 100
            batch_size = 100
            print(f"Starting to upsert values in pinecone in batch sizes of {batch_size}")
            for i in range(0, len(upsert_payload), batch_size):
                batch = upsert_payload[i:i+batch_size]
                self.index.upsert(
                    vectors=batch,
                    namespace=self.namespace,
                )
                print(f"Successfully upserted records {i} to {i+len(batch)}")
            print('\n\nAll vectors uploaded successfully')

        except Exception as e:
            print(f"Error in upserting the repository chunks to pinecone\n\n{e}")
            raise RuntimeError(f"Error in upserting the repository chunks to pinecone\n\n{e}")