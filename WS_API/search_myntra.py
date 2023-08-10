from bs4 import BeautifulSoup
import requests
from urllib.parse import quote

class MyntraScraper:
    def get_product_links(self, search_query):
        encoded_20 = quote(search_query)
        encoded_hy = '-'.join(quote(part) for part in search_query.split())
        url = f"https://www.myntra.com/{encoded_hy}?rawQuery={encoded_20}"
        print(url)

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
        response = requests.get(url, headers=headers, timeout=15)

        response = requests.get(url)
        # response = requests.get("https://www.google.com")
        print(response.status_code)

        soup = BeautifulSoup(response.content, 'html.parser')
        
        links = []
        ul_results = soup.find("ul", {"class": "results-base"})
        if ul_results:
            for li_element in ul_results.find_all("li", {"class": "product-base"}):
                link_element = li_element.find("a")
                if link_element and "href" in link_element.attrs:
                    links.append(link_element["href"])
                    print(link_element)
        return links

    def scrape_product_info(self, link):
        response = requests.get(link)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        metadata = {}
        metadata['name'] = soup.find("div", {"class": "pdp-name"}).get_text()
        metadata['price'] = soup.find("span", {"class": "pdp-price"}).find("strong").get_text()
        metadata['image_urls'] = []

        product_details_section = soup.find("div", {"class": "pdp-product-description-content"})
        if product_details_section:
            product_details = product_details_section.get_text()
            metadata['product_details'] = product_details

        size_fit_section = soup.find("div", {"class": "pdp-sizeFitDescContent"})
        if size_fit_section:
            size_fit = size_fit_section.get_text()
            metadata['size_fit'] = size_fit

        specifications_section = soup.find("div", {"class": "index-tableContainer"})
        if specifications_section:
            specifications = {}
            for index_row in specifications_section.find_all("div", {"class": "index-row"}):
                key = index_row.find("div", {"class": "index-rowKey"}).get_text()
                value = index_row.find("div", {"class": "index-rowValue"}).get_text()
                specifications[key] = value
            metadata['specifications'] = specifications

        for image_tag in soup.find_all("div", {"class": "image-grid-image"}):
            style_attr = image_tag.get("style")
            if style_attr and "background-image" in style_attr:
                image_url = style_attr.split("(")[1].split(")")[0].replace("&quot;", "")
                metadata['image_urls'].append(image_url)

        return metadata

def main():
    search_query = input("Enter search query: ")
    
    scraper = MyntraScraper()

    product_links = scraper.get_product_links(search_query)
    for link in product_links[:2]:
        metadata = scraper.scrape_product_info(link)
        print(metadata)

if __name__ == "__main__":
    main()
