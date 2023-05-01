from flask import Flask,request,render_template,jsonify
from src.pipelines.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

@app.route("/")
def home_page():
    return render_template("home.html")

@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("form.html")
    else:
            data=CustomData(
                Age=int(request.form.get('Age')),
                fnlwgt=int(request.form.get('fnlwgt')),
                education_num=int(request.form.get('education-num' )),
                capital_gain =int(request.form.get('capital-gain')),
                capital_loss =int(request.form.get('capital-loss')),
                hours_per_week=int(request.form.get('hours-per-week')),
                workclass= request.form.get('workclass'),
                education= request.form.get('education'),
                marital_status=request.form.get('marital-status'),
                occupation=request.form.get('occupation'),
                relationship=request.form.get('relationship'),
                sex=request.form.get('sex'),
                race=request.form.get('race'),
                native_country=request.form.get('native-country'),
                makes_over=request.form.get('makes over')
            )
            final_new_data=data.get_data_as_dataframe()
            predict_pipeline=PredictPipeline()
            pred=predict_pipeline.predict(final_new_data)

            results=round(pred[0],2)

            return render_template('result.html',final_result=results)


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)