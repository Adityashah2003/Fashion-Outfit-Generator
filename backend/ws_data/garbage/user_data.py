import requests
from bs4 import BeautifulSoup
import webbrowser
import pyautogui
import time

url = "https://www.flipkart.com/account/orders/search?order_time=2019%2CLast_30_days%2C2022%2C2021%2C2020%2COlder"

def get_order_links(order_page):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    response = requests.get(order_page, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    order_links = soup.find_all("a", class_="_3dshlq")  
    
    for order_link in order_links:
        href_value = order_link["href"]
        print("Found order link:", href_value)

def open_url_in_new_tab(url):
    webbrowser.open_new_tab(url)
    pyautogui.hotkey("ctrl", "t")  

def main():
    open_url_in_new_tab(url)
    time.sleep(5)
    get_order_links(url)

if __name__ == "__main__":
    main()
