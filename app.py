import streamlit as st
import cv2
import analysis as a
import ml as m

remove_default_footer = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(remove_default_footer, unsafe_allow_html=True)

st.title('Machine Learning Solution in Medical')


@st.cache
def load_img(img):
    image = cv2.imread(img)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def main():
    menu = ['Home', 'Data Analysis', 'Machine Learning', 'About']
    choice = st.sidebar.selectbox('', menu)

    if choice == 'Home':
        st.image(load_img('breastCancer.jpg'), use_column_width=True)

    elif choice == 'Data Analysis':
        a.analysis()

    elif choice == 'Machine Learning':
        m.machineLearning()

    else:
        pass


if __name__ == '__main__':
    main()
