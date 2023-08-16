from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)
CORS(app) 

@app.route('/recvprompt', methods=['POST'])

def handle_user_input():
    try:
        data = request.json  
        if 'message' in data:
            user_message = data['message']
            print("Received user message:", user_message) 

            response = {'response': user_message}  
            return jsonify(response), 200
        else:
            return jsonify({'error': 'Invalid request'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run()
