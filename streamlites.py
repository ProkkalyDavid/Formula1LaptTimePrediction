#Importáljuk a szükséges csomagokat
import streamlit as st
from joblib import load
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

#Betöltjük a modelleket
lin_reg_model = load('models/Formula1_LinearRegressionModel.joblib')
random_forest_model = load('models/Formula1_RandomForestModel.joblib')
decision_tree_model = load('models/Formula1_DecisionTreeModel.joblib')

#Létrehozzuk a Streamlit alkalmazást
st.title('Formula 1-es köridő predikció')

##Az input mező létrehozása
feature_value = st.sidebar.slider('Lap Number', 0.0, 10.0, 5.0)

#Gomb a predikció indításához
run_prediction = st.button('Predikció indítása')

#Ha a gombot megnyomták, lefuttatjuk a predikciókat
if run_prediction:
    # Modellek futtatása
    prediction_lin_reg = lin_reg_model.predict(np.array(input_data).reshape(1, -1))
    prediction_random_forest = random_forest_model.predict(np.array(input_data).reshape(1, -1))
    prediction_decision_tree = decision_tree_model.predict(np.array(input_data).reshape(1, -1))

    #Eredmények kiírása
    st.write('Lineáris regresszió predikciója: ', prediction_lin_reg[0])
    st.write('Random Forest predikciója: ', prediction_random_forest[0])
    st.write('Decision Tree predikciója: ', prediction_decision_tree[0])