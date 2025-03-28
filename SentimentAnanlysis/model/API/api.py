from flask import Flask, url_for, redirect , request, render_template, make_response
import pickle
from sklearn.pipeline import Pipeline

app = Flask(__name__)

models_file = open('c:/Users/Jharna Yadav/Documents/Internship/TCS/SentimentAnanlysis/model/API/models.pkl', 'rb')
model = pickle.load(models_file)

text_trans_file = open('C:/Users/Jharna Yadav/Documents/Internship/TCS/SentimentAnanlysis/model/API/text_trans.pkl', 'rb')
text_trans = pickle.load(text_trans_file)



tweet = "These rays reflects back in the sky."
sentiment = ["positive", "negative"]

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods = ['POST'])
def predict():
    tweet = request.form['tweet']
    pipeline = Pipeline([("count_vect", text_trans)])
    tweet_vectorized = pipeline.transform([tweet])
    pred1 = model[0].predict(tweet_vectorized)
    pred2 = model[1].predict(tweet_vectorized)
    pred3 = model[2].predict(tweet_vectorized)
    prediction = [pred1, pred2, pred3]
    zero = prediction.count(0)
    one = prediction.count(1)
    if zero<one :
        sent = sentiment[0]
    else:
        sent = sentiment[1]
    print(sent)
    return sent
    
if __name__ == "__main__": 
  app.run(debug = True)