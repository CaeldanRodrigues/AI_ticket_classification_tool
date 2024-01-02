import streamlit as st

from pages.admin_utlis import *


def main():
    st.set_page_config(page_title="Upload PDF to Pinecone Vector Store")
    st.title("Please upload your files...")

    pdf = st.file_uploader("Only PDF files allowed", type=["pdf"])

    if pdf is not None:
        with st.spinner('uploading...'):
            st.write("👉Reading PDF")
            text=read_pdf_data(pdf)

            st.write("👉Splitting data into chunks")
            docs_chunks=split_data(text)
            #st.write(docs_chunks)

            st.write("👉Creating embeddings instance")
            embeddings=create_embeddings_load_data()

            push_to_pinecone("gcp-starter","test-index",embeddings,docs_chunks)

        st.success("Successfully pushed the embeddings to Pinecone")



if __name__ == '__main__':
    main()
