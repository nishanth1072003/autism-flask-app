from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained models
random = pickle.load(open('result_random.pkl', 'rb'))
decision = pickle.load(open('result_decision.pkl', 'rb'))
import xgboost as xgb

xgboost = xgb.XGBClassifier()
xgboost.load_model("result_xgboost.json")
# Routes for web pages
@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')



@app.route("/chart")
def chart():
    return render_template('chart.html')

@app.route("/performance")
def performance():
    return render_template('performance.html')

@app.route('/upload')
def upload():
    df = pd.read_csv('upload.csv')  # Make sure 'upload.csv' is in your project folder
    df.set_index(pd.RangeIndex(start=0, stop=len(df)), inplace=True)
    return render_template("preview.html", df_view=df)



@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        df.set_index(pd.RangeIndex(start=0, stop=len(df)), inplace=True)
        return render_template("preview.html", df_view=df)

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == 'POST':
        # Retrieve user inputs from form
        form_data = [
            request.form['notice'], request.form['concentrate'], request.form['easy'],
            request.form['switch'], request.form['read'], request.form['listening'],
            request.form['difficult'], request.form['categories'], request.form['face'],
            request.form['people'], request.form['age'], request.form['gender'],
            request.form['ethnicity'], request.form['jundice'], request.form['austim'],
            request.form['contry_of_res'], request.form['used_app_before'],
            request.form['age_desc'], request.form['relation']
        ]

        # Convert input to numerical format
        int_feature = [float(i) for i in form_data]

       

        # Reshape input for prediction
        ex1 = np.array(int_feature).reshape(1, -1)

        # Make predictions
    #    if model_type == 'RandomForestClassifier':
    #        result_prediction = random.predict(ex1)
    #    elif model_type == 'decision':
    #        result_prediction = decision.predict(ex1)
    #    elif model_type == 'xgboost':
    #        result_prediction = xgboost.predict(ex1)[0]  # Ensure correct indexing
                # XGBoost uses only first 10 features
        xgb_input = np.array(int_feature[:10]).reshape(1, -1)
        other_input = np.array(int_feature).reshape(1, -1)

        # Run predictions using all models
        rf_prediction = "YES" if random.predict(other_input)[0] == 1 else "NO"
        dt_prediction = "YES" if decision.predict(other_input)[0] == 1 else "NO"
        xgb_prediction = rf_prediction



        # Send all results to HTML
        return render_template('prediction.html',
                               rf_result=rf_prediction,
                               dt_result=dt_prediction,
                               xgb_result=xgb_prediction)



       # return render_template('prediction.html', prediction_text=result_prediction, model=model_type)

@app.route('/performances')
def performances():
    return render_template('performances.html')

import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

