from flask import Flask, render_template, request
import numpy as np
from logging import debug
import pickle

# For Model
model = pickle.load(open('model_ETC.pkl', 'rb'))

app = Flask(__name__, template_folder="templates")


@app.route("/")
def main():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    age = float(request.form["age"])
    anaemia = float(request.form["anaemia"])
    creatinine_phosphokinase = float(request.form["creatinine_phosphokinase"])
    diabetes = float(request.form["diabetes"])
    ejection_fraction  = float(request.form["ejection_fraction"])
    high_blood_pressure = float(request.form["high_blood_pressure"])
    platelets = float(request.form["platelets"])
    serum_creatinine = float(request.form["serum_creatinine"])
    serum_sodium = float(request.form["serum_sodium"])
    sex = float(request.form["sex"])
    smoking = float(request.form["smoking"])
    time = float(request.form["time"])

    float_feature = [
        age,
        anaemia,
        creatinine_phosphokinase,
        diabetes,
        ejection_fraction,
        high_blood_pressure,
        platelets,
        serum_creatinine,
        serum_sodium,
        sex,
        smoking,
        time
    ]

    final_feature = [np.array(float_feature)]
    prediction = model.predict(final_feature)

    output = {
        0: "Tidak Meninggal",
        1: "Meninggal"
    }

    return render_template("index.html", prediction_text="Patients with heart failure have : {}".format(output[prediction[0]]))


if __name__ == '__main__':
    app.run(debug=True)
