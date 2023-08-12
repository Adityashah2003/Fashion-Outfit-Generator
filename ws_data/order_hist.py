import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

with open('C:\\Users\\Aditya\\Documents\\GitHub\\FOG\\data\\user_purchase_data.json', 'r') as json_file:
    user_data = json.load(json_file)

product_urls = []

for user in user_data:
    for purchase_history in user.get('purchase_history', []):
        if isinstance(purchase_history, str):
            product_urls.append(purchase_history)

def get_product_info(product_links):
    product_info_list = []
    
    for link in product_links:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, "html.parser")

        product_name = soup.find("span", {"class": "B_NuCI"}).get_text()
        product_price = soup.find("div", {"class": "_30jeq3 _16Jk6d"}).get_text()
        product_image = soup.find("img", {"class": "_2r_T1I _396QI4"})["src"]
        product_other_info = soup.find("div",{"class":"_1AN87F"}).get_text()

        specifications={}
        details_div = soup.find("div", {"class": "X3BRps"})
        rows = details_div.find_all("div", {"class": "row"})
        for row in rows:
            cols = row.find_all("div", {"class": "col"})
            if len(cols) == 2:
                key = cols[0].get_text(strip=True)
                value = cols[1].get_text(strip=True)
                specifications[key] = value

        product_info = {}
        product_info['name'] = product_name
        product_info['price'] =product_price
        product_info['image'] = product_image
        product_info['specs'] = specifications
        product_info['others'] = product_other_info
        
        product_info_list.append(product_info)
    
    return product_info_list

def main():
    all_product_info = [] 
    
    for url in product_urls:
        product_info = get_product_info([url])  # Pass the URL as a list to the function
        all_product_info.extend(product_info)    # Extend the list with the results from each URL

    for product in all_product_info:
        print(product)

if __name__ == "__main__":
    main()
