import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from flask import Flask, request, jsonify
from flask_cors import CORS
from research.ask_query import answer_user_question

app = Flask(__name__)
CORS(app)  # Her yerden gelen isteklere izin verir

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')
    try:
        response = answer_user_question(user_message)
    except Exception as e:
        print("Hata:", e)  # Hata mesajını terminale yaz
        response = "Bir hata oluştu. Lütfen tekrar deneyin."
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(port=5000)
