from PIL import Image
from sentence_transformers import SentenceTransformer
from lib.semantic_search import cosine_similarity
from lib.search_utils import load_movies
class MultimodalSearch:
    def __init__(self,documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.text = [] 
        for doc in documents:
            self.text.append(f"{doc['title']}: {doc['description']}")

        self.text_embeddings = self.model.encode(self.text, show_progress_bar= True)

    def search_with_image(self,image_path):
        image_embeddings = self.embed_image(image_path)
        similarity_score = []
        for idx,text_embed in enumerate(self.text_embeddings) :
            cosine_score = cosine_similarity(image_embeddings,text_embed)
            similarity_score.append({
                'number' : idx,
                'score' : float(cosine_score)
            })
        similarity_score = sorted(similarity_score , key = lambda x : x['score'],reverse=True)[:5]
        result = []
        for s in range(len(similarity_score)):
            Current_index = similarity_score[s]['number']
            result.append({
                'doc_id': self.documents[Current_index]['id'],
                'title' : self.documents[Current_index]['title'],
                'description' : self.documents[Current_index]['description'][:100],
                'score': similarity_score[s]['score']}
                )

        for idx,_ in enumerate(result):
            print(f'{idx+1}. {_['title']} (similarity: {_['score']:.3f})\n'
                  f'{_['description']}...')
             

    def embed_image(self,image_path):
        img = Image.open(image_path)
        return self.model.encode([img])[0]

    
def verify_emb(image_path):
    mm = MultimodalSearch()
    embedding = mm.embed_image(image_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(image_path):
    data = load_movies()
    MM = MultimodalSearch(data)
    return MM.search_with_image(image_path)