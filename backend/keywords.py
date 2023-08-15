from dotenv import load_dotenv
from ws_data import sort_instagram,sort_pinterest,order_hist
import openai
import os
import requests
import json

url = "https://api.openai.com/v1/completions"
# def chat_with_bot(prompt):
#     response = openai.Completion.create(
#         engine="davinci", 
#         prompt=prompt,
#         max_tokens= 100
#     )
#     return response.choices[0].text.strip()

def main():
    insta_data = sort_instagram.main()
    pin_data = sort_pinterest.main()
    user_data = order_hist.main()
    combined_prompt = f"{insta_data}\n{pin_data}\n{user_data}"
    # print(combined_prompt)

    load_dotenv()
    api_key = os.getenv('API_KEY')
    openai.api_key = api_key

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        # "model": "text-davinci-003",
        "model": "text-curie-001",
        "prompt": f"Extract keywords from this paragraph which make up a good fashion outfit: \"{combined_prompt}\"",
        "max_tokens": 40
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_json = response.json()
    generated_text = response_json["choices"][0]["text"]
    # print(generated_text)
    

if __name__ == "__main__":
    main()

