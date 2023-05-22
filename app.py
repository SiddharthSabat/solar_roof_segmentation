import streamlit as st

def main():
    st.title("Welcome to My Streamlit App")
    name = st.text_input("Enter your name:")
    if st.button("Submit"):
        if name:
            st.success(f"Hello, {name}!")
        else:
            st.warning("Please enter your name.")

if __name__ == "__main__":
    main()
