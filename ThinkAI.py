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
  tfidf_matrix = load_npz("ressources/matrix_NLP_SPACY.npz")
  with open("ressources/paragraphs_LEGAL_ARTICLES.pkl", "rb") as f:
    paragraphs = pickle.load(f)
  vectorizer = TfidfVectorizer()
  tfidf_matrix = vectorizer.fit_transform(paragraphs)
  max_tokens = 5
  input_tokens = input_sentence.split()[:max_tokens]
  input_tfidf = vectorizer.transform([' '.join(input_tokens)])
  similarities = cosine_similarity(input_tfidf, tfidf_matrix)
  indices = np.argsort(similarities, axis=1)[:, -2:]
  Sentences_NLP = [paragraphs[i] for i in indices[0]]
  return (Sentences_NLP)



def Final_texts(input_sentence):
  sentences = intitialize(input_sentence)
  models_result = MiniLM(input_sentence, sentences) + NLP(input_sentence, sentences)
  models_checkpoints = {"BertForQuestionAnswering": "bert-large-uncased-whole-word-masking-finetuned-squad"}
  pipe = pipeline("question-answering", model=models_checkpoints["BertForQuestionAnswering"])
  question = input_sentence
  answers = []
  for p in tqdm(models_result):
      if p:
          answer = pipe(question=question, context=p)
          answer["paragraph"] = p
          answers.append(answer)

  top_answers = sorted(answers, key=lambda x: x["score"], reverse=True)[:2]

  return (top_answers[0]['paragraph'], top_answers[1]['paragraph'])

def generate_response(prompt):
    # documents = Final_texts(prompt)

    # return documents[0] +"\n" +  documents[1]
    return "hello"