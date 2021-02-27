import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt
import seaborn as sns
import plotly.express as px


def analysis():
    df = pd.read_csv('data/data.csv')
    df1 = df.iloc[:, 1:-1]
    st.header('DataSet')
    st.dataframe(df.iloc[:, :-1])

    c1, c2 = st.beta_columns([3, 1])
    with c1:
        # pandas describe
        # =================
        st.header('Data Insight')
        st.dataframe(df1.describe().transpose())

    with c2:
        st.header('Concern')
        st.json({
            "M": "Cancer",
            "B": "No Cancer"
        })

    # pie chart
    # ==========
    with st.beta_expander('Patient %'):
        fig = px.pie(names=df['diagnosis'], title='Malignant vs. Benign', hole=0.4)
        fig.update_traces(textfont_size=15, textinfo='percent+label', textposition='inside')
        st.plotly_chart(fig, use_container_width=True)

    # scatter plot
    # =============
    with st.beta_expander('Plotting'):
        fig = px.scatter(data_frame=df1, x='area_mean', y='smoothness_mean', color='diagnosis',
                         title='Area mean vs. Smoothless mean', )
        st.plotly_chart(fig, use_container_width=True)

    # pair plot
    # ===========
    with st.beta_expander('Diagnostic Report Lookup'):
        fig = sns.pairplot(df1, hue='diagnosis',
                           vars=['radius_mean', 'texture_mean', 'area_mean', 'perimeter_mean', 'smoothness_mean'])
        st.pyplot(fig, use_container_width=True)

    # co-relation
    # ==========
    with st.beta_expander('Correlation'):
        fig = pt.figure(figsize=(17, 7))
        sns.heatmap(df1.corr(), annot=True, fmt='.1f')
        st.pyplot(fig)
