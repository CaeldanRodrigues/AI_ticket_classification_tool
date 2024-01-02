import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


def pull_from_pinecone(pinecone_environment,pinecone_index_name,embeddings):

    pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=pinecone_environment
    )

    index_name = pinecone_index_name

    index = Pinecone.from_existing_index(index_name, embeddings)
    return index

def create_embeddings():
    embeddings = OpenAIEmbeddings()
    # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

def get_similar_docs(index,query,k=2):

    similar_docs = index.similarity_search(query, k=k)
    return similar_docs

def get_answer(docs,user_input):
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_input)
    return response
