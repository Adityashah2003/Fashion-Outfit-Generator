from ws_data import search_flipkart
from flask import Flask, request,jsonify
from flask_cors import CORS  
import openai
import os
from dotenv import load_dotenv
import json
import user_input


app = Flask(__name__)
CORS(app, resources={r"/recvprompt": {"origins": "http://localhost:3000"},
                     r"/get_data": {"origins": "http://localhost:3000"}})

url = "https://api.openai.com/v1/completions"
load_dotenv()
api_key = os.getenv('API_KEY')
openai.api_key = api_key
    
@app.route('/recvprompt', methods=['POST'])
def handle_user_input():
    try:
        global ai_response
        data = request.json  
        if 'message' in data:
            user_message = data['message']
            ai_response = hold_conversation(user_message)  
            print("AI Response: ",ai_response)
            return jsonify({'response': ai_response}), 200 
        else:
            return jsonify({'error': 'Invalid request'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def hold_conversation(user_message):
    user_messages = []  
    ai_messages = []  
    user_messages.append(user_message) 
    ai_input = "\n".join(user_messages + ai_messages)  
    ai_reply = generate_response(ai_input)
    ai_messages.append(ai_reply)  
    return ai_reply


def generate_response(prompt, model="curie:ft-personal:curie-2-2023-08-13-08-22-41", max_tokens=70):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        # stop=["."]
    )
    return response.choices[0].text.strip()

@app.route('/get_data', methods=['GET'])
def get_data():
    # try:
    global ai_response
    extracted_keywords = user_input.main(ai_response)
    product_info_list = []
    for keyword in extracted_keywords:
        print("Keyword: ",keyword)
        product_info = search_flipkart.main(keyword)
        if product_info: 
            product_info_list.append({
                "name": keyword,
                "image": product_info["image"]["src"],
                "links": product_info["links"]
            })

    json_data = json.dumps(product_info_list, indent=2)
    return json_data, 200, {'Content-Type': 'application/json'}
app.run(port=5000)
