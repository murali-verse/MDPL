# 🏏 IPL 2025 Winner Prediction – MDPL Competition
Thiagarajar School of Management

## 📌 Project Title
Predicting the Winners of IPL 2025 Season using Machine Learning

## 🧠 About the Competition
This project was developed as part of the MDPL (Model Development and Predictive Learning) competition organized by Thiagarajar School of Management. The objective is to apply data science and machine learning techniques to accurately predict the winner and top-ranking teams of the Indian Premier League (IPL) 2025 season.

## 🧾 Methodology
### Data Collection
Historical team-level IPL statistics from 2008 to 2024, including batting, bowling, and results.

### Feature Engineering
1)Aggregated team-wise season stats

2)Polynomial feature expansion (up to degree 2)

3)Win ratio, average run rate, wickets taken, net run rate, etc.

4)Encoded categorical variables (team names, seasons)

### Model Used

1)Neural Network (Multilayer Perceptron)

2)Applied K-Fold Cross-Validation to prevent overfitting

3)Used Ordinal Regression logic to rank teams from 1st to 10th

### Evaluation

1)Manual validation using 2023 and 2024 outcomes

2)Metrics: Accuracy for Top 4, Rank Correlation (Spearman)

## 🛠️ Technologies Used
1)Python

2)Pandas, NumPy

3)Scikit-learn

4)TensorFlow/Keras or PyTorch (based on model choice)

5)Matplotlib, Seaborn

6)Jupyter Notebook
