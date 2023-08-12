import json

def main():
    json_file_path = 'C:\\Users\\Aditya\\Documents\\GitHub\\FOG\\data\\pin_data.json'

    with open(json_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    sorted_data = sorted(data, key=lambda x: x["aggregated_pin_data"]["saves"], reverse=True)
    top_2_grid_descriptions = [item["grid_description"] for item in sorted_data[:2]]

    top_descriptions_paragraph = "\n".join(top_2_grid_descriptions)
    return top_descriptions_paragraph

if __name__ == "__main__":
    top_descriptions = main()
