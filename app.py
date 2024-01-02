import streamlit as st

from user_utils import *

def main():

    st.header("Automatic Ticket Classification Tool")
    #Capture user input
    st.write("We are here to help you, please ask your question:")
    user_input = st.text_input("🔍")

    if user_input:
        embeddings=create_embeddings()
        index=pull_from_pinecone("gcp-starter","test-index",embeddings)
        relavant_docs=get_similar_docs(index,user_input)

        response=get_answer(relavant_docs,user_input)
        st.write(response)

        button = st.button("Submit ticket?")


if __name__ == '__main__':
    main()
