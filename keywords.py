from dotenv import load_dotenv
from ws_data import sort_instagram,sort_pinterest
import openai
import os
import requests
import json

url = "https://api.openai.com/v1/completions"

def chat_with_bot(prompt):
    response = openai.Completion.create(
        engine="davinci", 
        prompt=prompt,
        max_tokens= 100
    )
    return response.choices[0].text.strip()

def main():
    insta_data = sort_instagram.main()
    pin_data = sort_pinterest.main()
    combined_prompt = f"{insta_data}\n{pin_data}"

    load_dotenv()
    api_key = os.getenv('API_KEY')
    openai.api_key = api_key

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }
    data = {
    "model": "text-davinci-003",
    "prompt": f"Extract fashion outfit items from this paragraph: \"{combined_prompt}\"",
    "max_tokens": 100        
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_json = response.json()
    generated_text = response_json["choices"][0]["text"]
    outfit_start = "Outfit items:"
    outfit_items_text = generated_text.split(outfit_start)[1]
    outfit_items = [item.strip() for item in outfit_items_text.split(",")]

    print(outfit_items)

if __name__ == "__main__":
    main()

    







