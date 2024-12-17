import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Declaring the teams
teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

# Declaring the venues where the matches are going to take place
cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Loading the machine learning model from a saved pickle file
@st.cache_resource
def load_model():
    try:
        return pickle.load(open('pipe.pkl', 'rb'))
    except FileNotFoundError:
        st.error("Model file 'pipe.pkl' not found. Please ensure it's uploaded.")
        return None

pipe = load_model()

# Set up the app's title
st.title('IPL Win Predictor')

# Setting up the layout with two columns
col1, col2 = st.columns(2)

# Creating a dropdown selector for the batting team
with col1:
    battingteam = st.selectbox('Select the batting team', sorted(teams))

# Creating a dropdown selector for the bowling team
with col2:
    bowlingteam = st.selectbox('Select the bowling team', sorted(teams))

# Creating a dropdown selector for the city where the match is being played
city = st.selectbox('Select the city where the match is being played', sorted(cities))

# Creating a numeric input for the target score using number_input method in Streamlit
target = int(st.number_input('Target', step=1))

# Setting up the layout with three columns
col3, col4, col5 = st.columns(3)

# Creating a numeric input for the current score
with col3:
    score = int(st.number_input('Score', step=1))

# Creating a numeric input for the number of overs completed
with col4:
    overs = int(st.number_input('Overs Completed', step=1))

# Creating a numeric input for the number of wickets fallen
with col5:
    wickets = int(st.number_input('Wickets Fallen', step=1))

# Checking for different match results based on the input provided
if score > target:
    st.write(battingteam, "won the match")
    
elif score == target - 1 and overs == 20:
    st.write("Match Drawn")
    
elif wickets == 10 and score < target - 1:
    st.write(bowlingteam, 'Won the match')
    
elif wickets == 10 and score == target - 1:
    st.write('Match tied')
    
elif battingteam == bowlingteam:
    st.write('To proceed, please select different teams because no match can be played between the same teams')

else:
    # Checking if the input values are valid or not
    if target >= 0 and target <= 300 and overs >= 0 and overs <= 20 and wickets <= 10 and wickets >= 0 and score >= 0:
        
        try:
            if st.button('Predict Probability'):
                
                # Calculating the number of runs left for the batting team to win
                runs_left = target - score 
                
                # Calculating the number of balls left 
                balls_left = 120 - (overs * 6)
                
                # Calculating the number of wickets left for the batting team
                wickets_left = 10 - wickets
                
                # Calculating the current Run-Rate of the batting team
                current_run_rate = score / overs if overs > 0 else 0
                
                # Calculating the Required Run-Rate for the batting team to win
                required_run_rate = (runs_left * 6) / balls_left if balls_left > 0 else 0
                
                # Creating a pandas DataFrame containing the user inputs
                input_df = pd.DataFrame(
                    {'batting_team': [battingteam], 
                     'bowling_team': [bowlingteam], 
                     'city': [city], 
                     'runs_left': [runs_left], 
                     'balls_left': [balls_left], 
                     'total_runs_x': [target], 
                     'cur_run_rate': [current_run_rate], 
                     'req_run_rate': [required_run_rate]})
                
                # Ensure columns are treated as strings if they are categorical
                input_df['batting_team'] = input_df['batting_team'].astype(str)
                input_df['bowling_team'] = input_df['bowling_team'].astype(str)
                input_df['city'] = input_df['city'].astype(str)
                
                # Ensure the input DataFrame has the same column order as the model's training data
                expected_columns = [
                    'batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 
                    'total_runs_x', 'cur_run_rate', 'req_run_rate'
                ]
                input_df = input_df[expected_columns]

                # Make prediction only if model is loaded successfully
                if pipe:
                    result = pipe.predict_proba(input_df)
                
                    # Extracting the likelihood of loss and win
                    loss_prob = result[0][0]
                    win_prob = result[0][1]
                    
                    # Displaying the predicted likelihood of winning and losing in percentage
                    st.header(f'{battingteam} - {round(win_prob * 100)}%')
                    st.header(f'{bowlingteam} - {round(loss_prob * 100)}%')
                else:
                    st.error("Model is not available for prediction.")

        except ZeroDivisionError:
            st.error("Please fill all the details correctly.")
    
    # Displaying an error message if the input is incorrect        
    else:
        st.error('There is something wrong with the input, please fill the correct details.')
