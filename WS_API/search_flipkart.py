import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

def get_product_link(link_elements):
    product_links = []
    for link_element in link_elements:
        product_links.append(link_element["href"])
    product_links = product_links[:2]
    return product_links

def get_product_info(product_links):
    product_info_list = []
    
    for link in product_links:
        product_url = f"https://www.flipkart.com{link}"
        response = requests.get(product_url)
        soup = BeautifulSoup(response.content, "html.parser")

        product_name = soup.find("span", {"class": "B_NuCI"}).get_text()
        product_price = soup.find("div", {"class": "_30jeq3 _16Jk6d"}).get_text()
        product_image = soup.find("img", {"class": "_396cs4 _2amPTt _3qGmMb"})["src"]

        product_info = {
            "name": product_name,
            "price": product_price,
            "image": product_image,
            "url": product_url,  
        }
        
        product_info_list.append(product_info)
    
    return product_info_list

def main():
    extracted_keywords = input()
    encoded_query = quote(extracted_keywords)
    url = f"https://www.flipkart.com/search?q={encoded_query}"
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    link_elements = soup.find_all("a", {"class": "_1fQZEK"})
    product_links = get_product_link(link_elements)

    product_info_list = get_product_info(product_links)
    
    for product_info in product_info_list:
        print(product_info)

if __name__ == "__main__":
    main()
