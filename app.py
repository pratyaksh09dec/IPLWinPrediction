import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="IPL Win Probability Predictor",
    layout="centered"
)

@st.cache_resource
def load_model():
    with open("IPL_Win_Probability_Model.pkl", "rb") as file:
        return pickle.load(file)

@st.cache_resource
def load_columns():
    with open("model_columns.pkl", "rb") as file:
        return pickle.load(file)

@st.cache_resource
def load_metadata():
    with open("metadata.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()
model_columns = load_columns()
metadata = load_metadata()

teams = metadata["batting_teams"]
venues = metadata["venues"]

st.title("IPL Win Probability Predictor")

st.write("---")

batting_team = st.selectbox(
    "Batting Team",
    teams
)

bowling_team = st.selectbox(
    "Bowling Team",
    [team for team in teams if team != batting_team]
)

venue = st.selectbox(
    "Select Venue",
    venues
)

current_score = st.number_input(
    "Current Score",
    min_value=0,
    max_value=300,
    value=100,
    step=1
)

wickets_out = st.number_input(
    "Wickets Out",
    min_value=0,
    max_value=9,
    value=2,
    step=1
)

overs_completed = st.number_input(
    "Overs Completed",
    min_value=0.0,
    max_value=20.0,
    value=10.0,
    step=0.1,
    format="%.1f"
)

target = st.number_input(
    "Target",
    min_value=1,
    max_value=300,
    value=180,
    step=1
)

runs_left = target - current_score

over = int(overs_completed)

ball = int(round((overs_completed - over) * 10))

balls_bowled = over * 6 + ball

balls_left = 120 - balls_bowled

wickets_left = 10 - wickets_out

if overs_completed == 0:
    crr = 0
else:
    crr = current_score / overs_completed

if balls_left == 0:
    rrr = 0
else:
    rrr = (runs_left * 6) / balls_left

predict_button = st.button("Find Win Probability")

if predict_button:

    input_data = pd.DataFrame(
        {
            "batting_team": [batting_team],
            "bowling_team": [bowling_team],
            "city": [venue],
            "runs_left": [runs_left],
            "balls_left": [balls_left],
            "wickets_left": [wickets_left],
            "target": [target],
            "crr": [crr],
            "rrr": [rrr]
        }
    )

    input_data = pd.get_dummies(
        input_data,
        columns=[
            "batting_team",
            "bowling_team",
            "city"
        ]
    )

    input_data = input_data.reindex(
        columns=model_columns,
        fill_value=0
    )

    probability = model.predict_proba(input_data)[0]

    batting_probability = probability[1] * 100

    bowling_probability = probability[0] * 100

    st.write("---")

    st.subheader("Prediction Result")

    st.write(
        f"**{batting_team} Win Probability:** {batting_probability:.2f}%"
    )

    st.write(
        f"**{bowling_team} Win Probability:** {bowling_probability:.2f}%"
    )

    st.write("")

    if batting_probability > bowling_probability:

        st.write(
            f"**{batting_team} is more likely to win the match.**"
        )

    elif bowling_probability > batting_probability:

        st.write(
            f"**{bowling_team} is more likely to win the match.**"
        )

    else:

        st.write(
            "**The match is evenly balanced.**"
        )