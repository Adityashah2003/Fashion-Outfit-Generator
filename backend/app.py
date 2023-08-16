from ws_data import search_flipkart
import keywords
import requests


def main():
    extracted_keywords = keywords.main()
    for keyword in extracted_keywords:
        print(f"Processing keyword: {keyword}")
        product_info = search_flipkart.main(keyword)
        print(product_info)
        print("\n")

if __name__ == "__main__":
    main()

