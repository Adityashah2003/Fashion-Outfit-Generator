import json

with open('C:\\Users\\Aditya\\Documents\\GitHub\\FOG\\data\\cvae_data_mock.json', 'r') as json_file:
        json_data = json.load(json_file)

total_string_count = 0
list_of_keyword_lists = []

for entry in json_data:
    keyword_lists = [
        entry["Input fashion outfits"],
        entry["user_data"],
        entry["social_trends"],
        entry["recommend_output"]
    ]
    flat_keywords = [keyword for sublist in keyword_lists for keyword in sublist]
    total_string_count += len(flat_keywords)
    list_of_keyword_lists.append(flat_keywords)

print("Total string elements:", total_string_count)
print("List of lists with keywords:")
for idx, keywords in enumerate(list_of_keyword_lists, start=1):
    print(f"List {idx}: {keywords}")