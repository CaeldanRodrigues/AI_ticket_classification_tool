import streamlit as st



def main():

    st.header("Automatic Ticket Classification Tool")
    #Capture user input
    st.write("We are here to help you, please ask your question:")
    user_input = st.text_input("🔍")

    if user_input:
        st.write("👉Processing your question")


if __name__ == '__main__':
    main()
