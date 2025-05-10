# app.py

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# Title
st.title("🏏 IPL Team Result Predictor")

# STEP 1: Load training data (hardcoded CSV)
@st.cache_data
def load_training_data():
    df = pd.read_csv("ipl.csv")
    return df

train_df = load_training_data()

# STEP 2: Preprocess training data
le = LabelEncoder()
train_df['Result'] = le.fit_transform(train_df['Result'])

drop_cols = ['Team', 'Year', 'Total Maiden Overs', 'Top_bowler_Avg']
train_df.drop(columns=drop_cols, inplace=True, errors='ignore')

X = train_df.drop('Result', axis=1)
y = train_df['Result']

# Feature scaling & SMOTE
scaler = StandardScaler()
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_accuracy = accuracy_score(y_test, model.predict(X_test))*100
st.success(f"✅ Model trained with accuracy: {train_accuracy:.2f}%")

# STEP 3: File upload from user
st.subheader("📤 Upload Test Data File (.csv or .xlsx)")

test_file = st.file_uploader("Choose a test dataset", type=["csv", "xlsx"])

def load_test_file(file):
    try:
        # Try reading as CSV first
        return pd.read_csv(file)
    except Exception:
        try:
            file.seek(0)  # Reset file pointer after failed read
            return pd.read_excel(file, engine='openpyxl')
        except Exception as e:
            st.error(f"❌ Failed to read file: {e}")
            return None

# STEP 4: Run predictions
if test_file:
    test_df = load_test_file(test_file)
    if test_df is not None:
        team_names = test_df['Team'].values
        try:
            test_df.drop(columns=drop_cols, inplace=True, errors='ignore')
            X_2025 = test_df.drop('Result', axis=1, errors='ignore')
            X_2025_scaled = scaler.transform(X_2025)

            predictions = model.predict(X_2025_scaled)
            probs = model.predict_proba(X_2025_scaled)

            win_index = list(le.classes_).index(1) if 1 in le.transform(le.classes_) else 0

            result_df = pd.DataFrame({
                'Team': team_names,
                'Predicted Result': le.inverse_transform(predictions),
                'Win Probability': probs[:, win_index]
            })

            top_4 = result_df.sort_values(by='Win Probability', ascending=False).head(4).reset_index(drop=True)

            # Styled output
            st.markdown("## 🏆 IPL 2025 Predictions (Ordinal Regression):")
            st.markdown(f"**Winner** - {top_4.iloc[0]['Team']}")
            st.markdown(f"**Runner** - {top_4.iloc[1]['Team']}")
            st.markdown(f"**2nd Runner** - {top_4.iloc[2]['Team']}")
            st.markdown(f"**3rd Runner** - {top_4.iloc[3]['Team']}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
