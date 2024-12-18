import streamlit as st
import pickle5 as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def add_sidebar():
    st.sidebar.header('Cell Nuclei Measurements')
    data = get_clean_data()

    slider_label = [
        ('Radius (mean)', 'radius_mean'),
        ('Texture (mean)', 'texture_mean'),
        ('Perimeter (mean)', 'perimeter_mean'),
        ('Area (mean)', 'area_mean'),
        ('Smoothness (mean)', 'smoothness_mean'),
        ('Compactness (mean)', 'compactness_mean'),
        ('Concavity (mean)', 'concavity_mean'),
        ('Concave points (mean)', 'concave points_mean'),
        ('Symmetry (mean)', 'symmetry_mean'),
        ('Fractal dimension (mean)', 'fractal_dimension_mean'),
        ('Radius (se)', 'radius_se'),
        ('Texture (se)', 'texture_se'),
        ('Perimeter (se)', 'perimeter_se'),
        ('Area (se)', 'area_se'),
        ('Smoothness (se)', 'smoothness_se'),
        ('Compactnes (se)', 'compactness_se'),
        ('Concavity (se)', 'concavity_se'),
        ('Concave points (se)', 'concave points_se'),
        ('Symmetry (se)', 'symmetry_se'),
        ('Fractal dimension (se)', 'fractal_dimension_se'),
        ('Radius (worst)', 'radius_worst'),
        ('Texture (worst)', 'texture_worst'),
        ('Perimeter (worst)', 'perimeter_worst'),
        ('Area (worst)', 'area_worst'),
        ('Smoothness (worst)', 'smoothness_worst'),
        ('Compactnes (worst)', 'compactness_worst'),
        ('Concavity (worst)', 'concavity_worst'),
        ('Concave points (worst)', 'concave points_worst'),
        ('Symmetry (worst)', 'symmetry_worst'),
        ('Fractal dimension (worst)', 'fractal_dimension_worst')]

    input_dict = {}
    for label, key in slider_label:
        input_dict[key]=st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float (data[key].max()),
            value=float(data[key].mean())
        )

    return input_dict

def get_clean_data():
    data = pd.read_csv('data/data.csv')

    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

    return data

def get_scaled_value(input_dict):
    data = get_clean_data()

    X = data.drop(['diagnosis'], axis=1)

    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_value(input_data)
    categories = ['Radius','Texture','Perimeter','Area','Smoothness',
                  'Compactness','Concavity','Concave Points','Symmetry',
                  'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']

        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'],
            input_data['area_se'], input_data['smoothness_se'], input_data['compactness_se'],
            input_data['concavity_se'], input_data['concave points_se'], input_data['symmetry_se'],
            input_data['fractal_dimension_se']

        ],
        theta=categories,
        fill='toself',
        name='Standar Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']

        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return  fig

def add_predictions(input_data):
    model = pickle.load(open('model/model.pkl', 'rb'))
    scaler = pickle.load(open('model/scaler.pkl','rb'))

    input_array = np.array(list(input_data.values())).reshape(1,-1)

    # st.write(input_array)
    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader(
        'Cell cluster prediction'
    )
    st.write('The cell cluster is:')
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)

    # st.write(model.predict_proba(input_array_scaled))
    st.write('Probability of being benign:', model.predict_proba(input_array_scaled)[0][0])
    st.write('Probability of being malicious:', model.predict_proba(input_array_scaled)[0][1])

    st.write('This app can assist mdedical professional in making a diagnosis, but should not be used as substitute for a professional diagnosis ')
    # st.write(prediction)

def main():
        data = get_clean_data()
        data= data.drop(['diagnosis'],axis=1)
        st.set_page_config(
            page_title='Breast Cancer Prediction',
            page_icon=":female doctor;",
            layout='wide',
            initial_sidebar_state='expanded'
        )

        # st.markdown("""
        # <div style="padding: 1rem; border: 2px solid black; border-radius: 0.5rem; background-color: #7E99AB;">
        #     <p>Inline styled box - this should always work!</p>
        # </div>
        # """, unsafe_allow_html=True)

        with open('assets/style.css') as f:
            st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
            # st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        # with open('assets/style.css') as f:
        #     css_content = f.read()
        #     st.write("CSS Content:", css_content)  # Debug: Show content in Streamlit
        #     st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


        # st.write ('Hello world, we will build  a streamlit app')
        # print ('hello')

        input_data = add_sidebar()
        # st.write (input_data)

        with st.container():
            st.title('Breast Cancer Predictor')
            st.write('Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whetther a breast mass is benign or malignant based on th measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar')
        col1, col2 = st.columns([4,1])

        with col1:
            radar_chart = get_radar_chart(input_data)
            st.plotly_chart(radar_chart)
            # st.write('this is column 1')
        with col2:
            add_predictions(input_data)

        st.write(data)


if __name__ == '__main__':
    main()