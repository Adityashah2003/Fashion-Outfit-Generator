
from bs4 import BeautifulSoup
import requests
from urllib.parse import quote
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time

class MyntraScraper:
    def __init__(self, headers):
        self.headers = headers
        self.session = requests.Session()  # Create a session
        self.driver = self._init_selenium()

    def _init_selenium(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode (no browser window)
        driver = webdriver.Chrome(options=chrome_options)
        return driver

    def get_product_links(self, search_query):
        encoded_20 = quote(search_query)
        encoded_hy = '-'.join(quote(part) for part in search_query.split())
        url = f"https://www.myntra.com/{encoded_hy}?rawQuery={encoded_20}"
        print(url)

        response = self.session.get(url, headers=self.headers, timeout=15)
        self.driver.get(url)
        time.sleep(5)  # Wait for the content to load (you might need to adjust this delay)
        page_source = self.driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        links = []
        ul_results = soup.find("ul", {"class": "results-base"})
        print(ul_results)
        if ul_results:
            for li_element in ul_results.find_all("li", {"class": "product-base"}):
                link_element = li_element.find("a")
                if link_element and "href" in link_element.attrs:
                    links.append(link_element["href"])
                    print(link_element)
                if len(links) >= 2:
                    break
        return links

    def scrape_product_info(self, link):
        self.driver.get(link)
        page_source = self.driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
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
    
    custom_user_agent = "YourCustomUserAgent"
    headers = {"User-Agent": custom_user_agent}
    scraper = MyntraScraper(headers)

    product_links = scraper.get_product_links(search_query)
    for link in product_links:
        metadata = scraper.scrape_product_info(link)
        print(metadata)

    # Close the Selenium driver
    scraper.driver.quit()

if __name__ == "__main__":
    main()