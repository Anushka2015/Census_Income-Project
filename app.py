from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    else:
        data = CustomData(
            Age=float(request.form.get('Age')),
            education_num =float(request.form.get('education-num')),
            capital_gain =float(request.form.get('capital-gain')),
            hours_per_week =float(request.form.get('hours-per-week')),
            workclass = request.form.get('workclass'),
            education= request.form.get('education'),
            marital_status = request.form.get('marital-status'),
            occupation = request.form.get('occupation'),
            relationship = request.form.get('relationship'),
            race = request.form.get('race'),
            sex = request.form.get('sex'),
            native_country=request.form.get('native-country')
        )

        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        result = pred[0]
        output =""
        if result == 0.0:
            output = "Less than 50K"
        else:
            output = "More than 50K"
        return render_template('results.html',final_result=output)




if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)