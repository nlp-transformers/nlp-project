import streamlit as st
from recorder import AudioRecorder

def main():
    st.set_page_config(page_title='Voice transformer')
    st.header('Transforming the world')

    rec = AudioRecorder()
    rec.record()


if __name__ == '__main__':
    main()