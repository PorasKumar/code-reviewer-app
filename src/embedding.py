from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
load_dotenv()
import os

class Embeddings:
    def __init__(self,dense_model, text_in_chunks:list[str]):

        if not text_in_chunks:
            raise ValueError("No Text in Chunks!")
        self.text_in_chunks = text_in_chunks

        #init HuggingFace
        self.dense_model = dense_model

        try:
            # Simple whitespace tokenization to avoid NLTK dependency conflicts
            #tokenise the texts into single single token, because rank_bm25 requires tokens to learn the vocabulary of each word
            self.tokenized_corpus = [doc.lower().split() for doc in self.text_in_chunks]
            self.bm25 = BM25Okapi(self.tokenized_corpus) #ek ek token par bm25 will work to learn vocab
        except Exception as e:
            print(f"Error in initialising BM25 sparse vector model: {e}")
            raise RuntimeError(f"Error in Initialising BM25 sparse vector embedding model\n{e}")


    #function for dense embeddings 
    def dense_embeddings_function(self):
        try:
            return self.dense_model.encode(self.text_in_chunks).tolist()
        
        except Exception as e:
            print(f"Error in creating dense embeddings using HuggingFace transformer \n{e}")
            raise RuntimeError(f"Error in creating dense embeddings using HuggingFace transformer \n{e}")

    #function for sparse embeddings
    def sparse_embeddings_function(self):
        try:
            sparse_vectors = []

            #loop through each document's tokens, chunk by chunk we iterate
            for doc_tokens in self.tokenized_corpus:
                indices = []
                values = []

                #getting unique tokens in this chunk to avoid duplicacy of data using set() func
                unique_tokens = set(doc_tokens)

                for token in unique_tokens:
                    if token in self.bm25.idf: #if token exist in the doc, prevents crashing if any changes made in docs
                        score = self.bm25.idf[token] #calculate the idf (rarity)
                        if (score>0):
                            #create a hash id for it, so that scaled query hash matches
                            token_id = abs(hash(token)) % (2**31-1)

                            indices.append(token_id)
                            values.append(float(score))

                sparse_vectors.append({
                    "indices":indices,
                    "values":values,
                })

            return sparse_vectors, self.bm25
        
        except Exception as e:
            print(f"Error in creating sparse embeddings using BM25 Sparse model \n{e}")
            raise RuntimeError(f"Error in creating sparse embeddings using BM25 Sparse model \n{e}")