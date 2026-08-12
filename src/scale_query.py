import numpy as np
from rank_bm25 import BM25Okapi


def scale_query_function(query:str ,dense_model, bm25, alpha):
    """Scaling the query into dense and sparse values for hybrid search"""

    #we are using the same bm25 here, which is already trained on our chunks, it has learnt the vocab of our chunks
    try:
        #dense scaling
        raw_dense = dense_model.encode(query).tolist()
        scaled_dense = [float(v*alpha) for v in raw_dense]

        #sparse vector
        unique_tokens = set(query.lower().split())  #unique so that we have only one set of indices and values for a keyword
        indices = []
        values = []

        for token in unique_tokens:
            if token in bm25.idf:
                #extract raw id score and scale using (1-alpha)
                raw_score = bm25.idf[token] #inverse doc frequency of each token
                scaled_score = float(raw_score * (1-alpha)) #scaled score of the token

                if (scaled_score>0):

                    #calculating a hash value for token_id 
                    #hash is a 32 bit signed value, so we use abs() to give only +ve
                    token_id = abs(hash(token)) % (2**31-1)

                    indices.append(token_id)
                    values.append(scaled_score)

        scaled_sparse = {
            "indices":indices,
            "values":values,
        }

        return scaled_dense, scaled_sparse

    except Exception as e:
        print(f"Exception in scaling query\n\n:{e}")
        raise RuntimeError(f"Exception in scaling query\n\n:{e}")