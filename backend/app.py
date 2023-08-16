from ws_data import search_flipkart
import keywords
import json

def main():
    extracted_keywords = keywords.main()
    product_info_list = []

    for keyword in extracted_keywords:
        product_info = search_flipkart.main(keyword)
        if product_info:  # Check if product_info is not None
            product_info_list.append({
                "name": product_info["name"],
                "image": product_info["image"]["src"]  # Extract the 'src' attribute
            })
    
    # Convert the product_info_list to JSON
    json_data = json.dumps(product_info_list, indent=2)
    return json_data

if __name__ == '__main__':
    main()
