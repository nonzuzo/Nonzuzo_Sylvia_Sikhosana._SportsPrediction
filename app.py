import numpy as np
import pandas as pd
import streamlit as st 
from sklearn import preprocessing
#import pickle
import joblib

model = joblib.load('ensemble_model.pkl')
encoder_dict =joblib.load('scaler_model.pkl')
cols=['movement_reactions', 'passing', 'mentality_composure', 'value_eur', 'wage_eur','dribbling', 'attacking_short_passing', 'mentality_vision', 'international_reputation','skill_long_passing']
  
def main(): 
    st.title("Overall Score of Player Predictor")
    html_temp = """
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Overall Score Prediction App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    
    movement_reactions = st.slider("Movement Reactions", 1, 100, 50)
    passing = st.slider("Passing", 1, 100, 50)
    mentality_composure = st.slider("Mentality Composure", 1, 100, 50)
    value_eur = st.slider("Value EUR", 1, 100, 50)
    wage_eur = st.slider("Wage EUR", 1, 100, 50)
    dribbling = st.slider("Dribbling", 1, 100, 50)
    attacking_short_passing = st.slider("Attacking Short Passing", 1, 100, 50)
    mentality_vision = st.slider("Mentality Vision", 1, 100, 50)
    international_reputation = st.slider("International Reputation", 1, 100, 50)
    skill_long_passing = st.slider("Skill Long Passing", 1, 100, 50)
     
     

    if st.button("Predict"):
        features = [[movement_reactions, passing, mentality_composure, value_eur, wage_eur,dribbling, attacking_short_passing, mentality_vision, international_reputation, skill_long_passing]]
    
    data = {
        'movement_reactions': int(movement_reactions),
        'passing': int(passing),
        'mentality_composure': int(mentality_composure),
        'value_eur': int(value_eur),
        'wage_eur': int(wage_eur),
        'dribbling': int(dribbling),
        'attacking_short_passing': int(attacking_short_passing),
        'mentality_vision': int(mentality_vision),
        'international_reputation': int(international_reputation),  # categorical
        'skill_long_passing': int(skill_long_passing)
    }

    # Convert data to DataFrame
    df = pd.DataFrame([list(data.values())], columns=list(data.keys()))
                

    features_list = df.values.tolist()           
    prediction = model.predict(features_list)
    print(features_list)
    
    st.write("Input features:")
    st.write(df)
    
    st.write(f"Prediction: {prediction}")
     
if __name__=='__main__': 
    main()