import xml.etree.ElementTree as ET
import os

# Count the number of each label in the dataset
all_possible = set()
label_counter = dict()
for dirpath, subdirname, subfilename in os.walk("dataset_fixed_capital_scrollable"):
    if subdirname == []:
        for subfile in subfilename:
            if subfile[-4:] == ".xml":
                file = dirpath + "\\" + subfile
                tree = ET.parse(file)
                root = tree.getroot()
                for single_object in root.findall('object'):
                    label_counter[single_object[0].text] = label_counter.get(single_object[0].text, 0) + 1
                    if single_object[0].text not in all_possible:
                        print(file)
                        print(single_object[0].text)
                        all_possible.add(single_object[0].text)

cnt = 0
for key, value in label_counter.items():
    cnt += value
    print(f"{key}: {value}")
    
print("total: ", cnt)
