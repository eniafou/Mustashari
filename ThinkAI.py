import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix, load_npz
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import chromadb
from chromadb.utils import embedding_functions
import fitz
from transformers import pipeline
from tqdm import tqdm
import random
import huggingface_hub
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


#Reading dataset
def intitialize(input_sentence):
  
  pdf_file = open('ressources/loi_english.pdf', 'rb')
  pdf_reader = PyPDF2.PdfReader(pdf_file)
  

  text_content = ''
  for page in range(len(pdf_reader.pages)):
      text_content += pdf_reader.pages[page].extract_text()

  # Split the text content into individual sentences
  sentences = text_content.split('.')

  return sentences

#MINILM Model
def MiniLM(input_sentence, sentences):
  model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device='cpu')
  sentence_embeddings = np.loadtxt('ressources/matrix_MINILM.csv', delimiter=',')
  input_embedding = model.encode([input_sentence])
  similarities = cosine_similarity(input_embedding, sentence_embeddings)[0]
  Sentences_MINILM = [s for _, s in sorted(zip(similarities, sentences), reverse=True)]
  Sentences_MINILM = Sentences_MINILM[:3]
  return(Sentences_MINILM)

#NLP_SPACY Model
def NLP(input_sentence, sentences):
  tfidf_matrix = load_npz("ressources/matrix_NLP_SPACY (2).npz")
  with open('ressources/vectorizer (3).pkl', 'rb') as file:
      vectorizer = pickle.load(file)
  with open("ressources/paragraphs_LEGAL_ARTICLES (3).pkl", "rb") as f:
      paragraphs = pickle.load(f)
  max_tokens = 5
  input_tokens = input_sentence.split()[:max_tokens]
  input_tfidf = vectorizer.transform([' '.join(input_tokens)])
  similarities = cosine_similarity(input_tfidf, tfidf_matrix)
  indices = np.argsort(similarities, axis=1)[:, -2:]
  Sentences_NLP = [paragraphs[i] for i in indices[0]]
  return (Sentences_NLP)




# def NLP2(input_sentence, sentences):
#   tfidf_matrix = np.load("ressources/matrix_NLP_SPACY.npy")
#   vectorizer = TfidfVectorizer()

#   with open('ressources/vectorizer.pickle', 'rb') as handle:
#     vectorizer = pickle.load(handle)
#   max_tokens = 5
#   input_tokens = input_sentence.split()[:max_tokens]
#   input_tfidf = vectorizer.transform([' '.join(input_tokens)])
#   similarities = cosine_similarity(input_tfidf, tfidf_matrix)
#   indices = np.argsort(similarities, axis=1)[:, -2:]
#   Sentences_NLP = [sentences[i] for i in indices[0]]
#   return (Sentences_NLP)

def Final_texts1(input_sentence):
  sentences = intitialize(input_sentence)
  models_result = MiniLM(input_sentence, sentences)[:2]
  return models_result

def Final_texts(input_sentence):
  sentences = intitialize(input_sentence)
  models_result = MiniLM(input_sentence, sentences) + NLP(input_sentence, sentences)
  models_checkpoints = {"BertForQuestionAnswering": "bert-large-uncased-whole-word-masking-finetuned-squad"}
  with open('ressources/pipeline.pkl', 'rb') as file:
    pipe = pickle.load(file)
  question = input_sentence
  answers = []
  for p in tqdm(models_result):
      if p:
          answer = pipe(question=question, context=p)
          answer["paragraph"] = p
          answers.append(answer)

  top_answers = sorted(answers, key=lambda x: x["score"], reverse=True)[:2]
  


def generate_response(prompt):
    
    prompt = from_darija(prompt)
    print(prompt)
    documents = Final_texts1(prompt)
    Message = f"Answer the question as a legal consultant using the documents provided bellow : {prompt} :\n"
    if documents!=None:

      return Message + documents[0] +"\n\n" +  documents[1]
    
    
    return "I didn't find anything, Sorry."
  
    # return "hello"



def from_darija(prompt):
  huggingface_hub.login( token='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')
  tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True, src_lang="ary_Arab")
  model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M", use_auth_token=True)
  Question = prompt #"شنو غادي يوقع ايلا شفت جريمة وتم ابتزازي"
  inputs = tokenizer(Question, return_tensors="pt")

  translated_tokens = model.generate(
    **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=40
  )
  input_sentence = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

  return input_sentence