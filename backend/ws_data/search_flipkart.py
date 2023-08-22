import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

def get_product_link(link_elements):
    product_links = []
    for link_element in link_elements:
        product_links.append(link_element["href"])
    product_links = product_links[:1]
    return product_links

def get_product_info(product_links):
    product_info_list = []
    
    for link in product_links:
        product_url = f"https://www.flipkart.com{link}"
        response = requests.get(product_url)
        soup = BeautifulSoup(response.content, "html.parser")

        product_name = soup.find("span", {"class": "B_NuCI"})
        if product_name:
            product_name = product_name.get_text()
        product_image = soup.find("img", {"class": "_2r_T1I _396QI4"})
        product_other_info = soup.find("div",{"class":"_1AN87F"})
        if product_other_info:
            product_other_info = product_other_info.get_text()

        specifications={}
        details_div = soup.find("div", {"class": "X3BRps"})
        if details_div:
            rows = details_div.find_all("div", {"class": "row"})
            for row in rows:
                cols = row.find_all("div", {"class": "col"})
                if len(cols) == 2:
                    key = cols[0].get_text(strip=True)
                    value = cols[1].get_text(strip=True)
                    specifications[key] = value

            product_info = {}
            product_info['name'] = product_name
            product_info['links'] = link
            product_info['image'] = product_image
            
            product_info_list.append(product_info)
    
    return product_info_list

def main(product):
    encoded_query = quote(product)
    url = f"https://www.flipkart.com/search?q={encoded_query}"
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    link_elements = soup.find_all("a", {"class": "_2UzuFa"})
    product_links = get_product_link(link_elements)

    product_info_list = get_product_info(product_links)
    for product_info in product_info_list:
        return product_info

if __name__ == "__main__":
    main()
