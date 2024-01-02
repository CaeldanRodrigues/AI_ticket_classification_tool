import streamlit as st
from dotenv import load_dotenv


def main():
    load_dotenv()
    st.set_page_config(page_title="Dump PDF to Pinecone - Vector Store")
    st.title("Please upload your files...📁 ")

    pdf = st.file_uploader("Only PDF files allowed", type=["pdf"])

    if pdf is not None:
        with st.spinner('Wait for it...'):
            # read pdf function
            st.write("👉Reading PDF done")

            # Create chunks
            st.write("👉Splitting data into chunks done")

            # Create the embeddings
            st.write("👉Creating embeddings instance done")

            # Build the vector store (Push the PDF data embeddings)

        st.success("Successfully uploaded the embeddings to Pinecone")


if __name__ == '__main__':
    main()
