# Mustashari

## Table of Contents
- [About the project](#about-the-project)
- [Team](#team)
- [Usage](#usage)
- [Architecture](#architecture)
- [Technology Used and Credits](#technology-used-and-credits)


## About the project 
Mustashari is an app that enables Moroccans to access cheap legal consulting. Our goal is for Mustashari to be able to take questions in Moroccan Darija and give an appropiate answer based on the relevant articles from the Moroccan law. Right now Mustashari is able to understand text in Darija and pair it with the appropriate sections from the morrocan law. However, it doesn't generate satisfying answers because we currently lack access to a robust generative model API. It's now a prompt generator that can be used with a powerful AI like ChatGPT.

## Team
-[Ayyoub El Kasmi](https://www.linkedin.com/in/ayyoub-el-kasmi-727578236/)
-[Soufiane Ait El Aouad](https://www.linkedin.com/in/soufiane-ait-el-aouad/)
-[Marouane Amaadour](https://www.linkedin.com/in/marouane-amaadour-6ab824229/)

## Usage
To use Mustashari on your machine you can …
- Step 1: clone the repo
- Step 2: run the command "pip install -r requirements"
- Step 4: run the command "streamlit run streamlit _app.py"


Or you can just click on [this link](https://eniafou-mustashari-streamlit-app-rw5r56.streamlit.app/) to use it on the web.

You can also look throughs the notebooks to better understand how the code works.
## Architecture
![Flowchart](./media/final_b.pngs)

## Difficulties and challenges
This was our first time working on a project about generative AI. We had to learn how to use APIs and combine multiple technologies from multiple sources to achieve a specific goal.
We had to learn about retrieval models, embeddings, cosine similarity, Hugging Face, prompt engineering ...

In the begenning we wanted to use the OpenAI API to generate the final answers, but we soon learned that it wasn't free. We decided to work with **cohere** as an alternative, however we learned later on that it wasn't a robust model or at least we weren't 
## Technology used and Credits
-[No language left behind (Meta)](https://ai.facebook.com/research/no-language-left-behind/)
-[miniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
-[spaCy](https://spacy.io/)
-[DocQuery](https://github.com/impira/docquery)
