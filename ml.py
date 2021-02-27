import streamlit as st
import joblib
import numpy as np
from pickle import load
import pandas as pd


def machineLearning():
    df = pd.read_csv('data/data.csv')  # load data
    df = df.iloc[:, :-1]  # skip unnecessary col
    df1 = df.iloc[:, 2:]  # genuine data
    st.header('DataSet')
    st.dataframe(df)

    model = load(open('breastCancerClassificationModel.pkl', 'rb'))  # load model
    sc = joblib.load('scaler.save')  # load scaling

    menu = ['Whole Data', 'Single Data']

    c1, c2 = st.beta_columns(2)
    c3, c4, c5 = st.beta_columns([2, 2, 1])

    with c1:
        st.header('Apply ML')
        choice = st.selectbox('', menu)
        process = st.button('Process')

    if choice == 'Whole Data' and process:

        with c3:
            with st.beta_expander('Malignant Or Benign'):
                test = sc.transform(np.array(df1))
                result = model.predict(test)
                result = ["Yes" if r == 'M' else "No" for r in result]
                id = pd.DataFrame(df.iloc[:, 0], columns=['id'])  # patient Id
                isCancer = pd.DataFrame(result, columns=['cancer'])
                result = pd.concat([id, isCancer], axis=1)
                st.dataframe(result)

        with c4:
            with st.beta_expander('Opinion'):
                st.radio('Result is Correct?', ['Yes', 'No'])
                st.button('Send')

    elif choice == 'Single Data':
        # All user input
        # ==============
        # Mean
        # ======
        with st.sidebar.beta_expander('Mean'):
            rm = st.number_input('Radius Mean', 0.0)
            tm = st.number_input('Texture Mean', 0.0)
            pm = st.number_input('Perimeter Mean', 0.0)
            am = st.number_input('Area Mean', 0.0)
            sm = st.number_input('Smoothness Mean', 0.0)
            cm = st.number_input('Compactness Mean', 0.0)
            cam = st.number_input('Concavity Mean', 0.0)
            cpm = st.number_input('Concave points Mean', 0.0)
            sym = st.number_input('Symmetry Mean', 0.0)
            fm = st.number_input('Fractal Dimension Mean', 0.0)

        # Se
        # ===
        with st.sidebar.beta_expander('Se'):
            rs = st.number_input('Radius se', 0.0)
            ts = st.number_input('Texture se', 0.0)
            ps = st.number_input('Perimeter se', 0.0)
            ars = st.number_input('Area se', 0.0)
            ss = st.number_input('Smoothness se', 0.0)
            cs = st.number_input('Compactness se', 0.0)
            cas = st.number_input('Concavity se', 0.0)
            cps = st.number_input('Concave points se', 0.0)
            sys = st.number_input('Symmetry se', 0.0)
            fds = st.number_input('Fractal Dimension se', 0.0)

        # Worst
        # ======
        with st.sidebar.beta_expander('Worst'):
            rw = st.number_input('Radius Worst', 0.0)
            tw = st.number_input('Texture Worst', 0.0)
            pw = st.number_input('Perimeter Worst', 0.0)
            aw = st.number_input('Area Worst', 0.0)
            sw = st.number_input('Smoothness Worst', 0.0)
            cw = st.number_input('Compactness Worst', 0.0)
            caw = st.number_input('Concavity Worst', 0.0)
            cpw = st.number_input('Concave points Worst', 0.0)
            syw = st.number_input('Symmetry Worst', 0.0)
            fdw = st.number_input('Fractal Dimension Worst', 0.0)

        input_data = {
            'Radius Mean': rm,
            'Texture Mean': tm,
            'Perimeter Mean': pm,
            'Area Mean': am,
            'Smoothness Mean': sm,
            'Compactness Mean': cm,
            'Concavity Mean': cam,
            'Concave points Mean': cpm,
            'Symmetry Mean': sym,
            'Fractal Dimension Mean': fm,

            'Radius Se': rs,
            'Texture Se': ts,
            'Perimeter Se': ps,
            'Area Se': ars,
            'Smoothness Se': ss,
            'Compactness Se': cs,
            'Concavity Se': cas,
            'Concave points Se': cps,
            'Symmetry Se': sys,
            'Fractal Dimension Se': fds,

            'Radius Worst': rw,
            'Texture Worst': tw,
            'Perimeter Worst': pw,
            'Area Worst': aw,
            'Smoothness Worst': sw,
            'Compactness Worst': cw,
            'Concavity Worst': caw,
            'Concave points Worst': cpw,
            'Symmetry Worst': syw,
            'Fractal Dimension Worst': fdw,
        }

        with c3:
            with st.beta_expander('Input Data', expanded=True):
                st.json(input_data)
        with c4:
            with st.beta_expander('Machine Predicts'):
                test1 = sc.transform(np.array([[rm, tm, pm, am, sm, cm, cam, cpm, sym, fm,
                                                rs, ts, ps, ars, ss, cs, cas, cps, sys, fds,
                                                rw, tw, pw, aw, sw, cw, caw, cpw, syw, fdw]]))
                result1 = model.predict(test1)
                accuracy = str(round(model.best_score_ * 100, 2)) + ' %'

                if result1[0] == "B":
                    st.json({
                        "Cancer": "No",
                        "Accuracy (Given Data)": accuracy
                    })
                else:
                    st.json({
                        "Accuracy (Given Data)": accuracy
                    })
        with c5:
            with st.beta_expander('Opinion', expanded=True):
                st.radio('Result is Correct?', ['Yes', 'No'])
                if st.button('Send'):
                    st.success('Result is sent to server.')
