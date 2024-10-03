from flask import Flask, request, render_template, jsonify
import pickle
import joblib
from urllib.parse import urlparse



app = Flask(__name__)
# CORS(app) 
def extract_features(url):
    parsed_url = urlparse(url)
    tld = parsed_url.hostname.split('.')[-1]
    
    hostname_length = len(parsed_url.hostname)
    path_length = len(parsed_url.path)
    fd_length = len(parsed_url.netloc)
    tld_length = len(tld)
    count_dash = parsed_url.path.count('-')
    count_at = parsed_url.path.count('@')
    count_question = parsed_url.path.count('?')
    count_percent = parsed_url.path.count('%')
    count_dot = parsed_url.path.count('.')
    count_equal = parsed_url.path.count('=')
    count_http = url.count('http')
    count_https = url.count('https')
    count_www = parsed_url.hostname.count('www')
    count_digits = sum(c.isdigit() for c in parsed_url.path)
    count_letters = sum(c.isalpha() for c in parsed_url.path)
    count_dir = url.count('/')
    use_of_ip = 1 if parsed_url.hostname.replace('.', '').isdigit() else 0
    
    return [hostname_length, path_length, fd_length, tld_length, count_dash,
            count_at, count_question, count_percent, count_dot, count_equal,
            count_http, count_https, count_www, count_digits, count_letters,
            count_dir, use_of_ip]

with open("models/spam_sms_model.pkl", "rb") as file:
    model = pickle.load(file)

with open('models/vectorizer', 'rb') as file:
    vectorizer = pickle.load(file)

urlmodel = joblib.load('models/logreg.pkl')

with open('models/vectorizer2', 'rb') as file:
    urlvectorizer = pickle.load(file)

with open('models/payment_fraud_model.pkl', 'rb') as file:
    payment_fraud_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/spam-sms')
def spam_sms_detection():
    return render_template('spam_sms_detection.html')

@app.route("/sms-predict", methods=["POST"])
def predict():
    message = request.form["message"]
    message = [message]
    new_message_vectorized = vectorizer.transform(message)
    prediction = model.predict(new_message_vectorized)
    result = "spam" if prediction[0] == 1 else "Not Spam"
    return jsonify({"prediction":result})

@app.route('/phishing-url')
def phishing_url_detection():
    return render_template('phishing_url_detection.html')

@app.route("/url-predict", methods=["POST"])
def urlpredict():
    message = request.form["message"]
    new_message_vectorized = extract_features(message)
    new_message_vectorized = [new_message_vectorized]
    prediction = urlmodel.predict(new_message_vectorized)
    result = "Bad URL" if prediction[0] == 1 else "Good URL"
    
    return jsonify({"prediction":result})

@app.route('/payment-fraud')
def payment_fraud_detection():
    return render_template('payment_fraud_detection.html')


@app.route('/payment-predict', methods=["POST"])
def payment_fraud_predict():
    if request.method == 'POST':
        try:
            features = [float(x) for x in request.form.values()]
            prediction = payment_fraud_model.predict([features])[0]
        except Exception as e:
            return str(e)
    return render_template('payment_fraud_detection.html', prediction=prediction)
    


if __name__ == '__main__':
    app.run(host='0.0.0.0')


