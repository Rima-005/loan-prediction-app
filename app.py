from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model/loan_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    age = int(request.form['age'])
    income = float(request.form['income'])
    loan_amount = float(request.form['loan_amount'])
    credit_score = int(request.form['credit_score'])

    prediction = model.predict([
        [age, income, loan_amount, credit_score]
    ])

    if prediction[0] == 1:
        result = "Loan Approved ✅"
    else:
        result = "Loan Rejected ❌"

    return render_template(
        'index.html',
        prediction_text=result
    )

if __name__ == "__main__":
    app.run(debug=True)