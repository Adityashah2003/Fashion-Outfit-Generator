from dotenv import load_dotenv
from ws_data import sort_instagram,sort_pinterest,order_hist
from auto_encoder import cvae_2

import openai
import os
import requests
import json
import re

url = "https://api.openai.com/v1/completions"

def remove_formatting(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\[\],]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    insta_data = sort_instagram.main()
    pin_data = sort_pinterest.main()
    user_data = order_hist.main()
    ae_kw = cvae_2.main()
    combined_prompt = f"{insta_data}{pin_data}{user_data}{ae_kw}"
    formatted_prompt = remove_formatting(combined_prompt)
    # print(formatted_prompt)

    load_dotenv()
    api_key = os.getenv('API_KEY')
    openai.api_key = api_key

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": "text-davinci-003",
        # "model": "text-curie-001",
        # "prompt": f"Extract keyowrds which make up a good fashion outfit when wore together: \"{formatted_prompt}\"",
        "prompt": f"Create a good fashion outfit from these {formatted_prompt} and list out its items.",
        "max_tokens": 40
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response_json = response.json()
    generated_text = response_json["choices"][0]["text"]
    # print(generated_text)
    output_lines = generated_text.split('\n')
    output_lines = [line.strip() for line in output_lines if line.strip() != '']

    item_names = []
    for line in output_lines:
        words_in_sentence = [word.strip() for word in line.split(',')]
        if len(words_in_sentence) > 1:
            item_names.extend(words_in_sentence)
        else:
            item_names.append(re.sub(r'^\d+\.\s*', '', line))
    return item_names
    

if __name__ == "__main__":
    main()

