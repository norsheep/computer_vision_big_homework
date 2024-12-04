import xml.etree.ElementTree as ET
import os

for dirpath, subdirname, subfilename in os.walk("dataset_fixed_capital_scrollable"):
    if subdirname == []:
        for subfile in subfilename:
            if subfile[-4:] == ".xml":
                file = dirpath + "\\" + subfile
                tree = ET.parse(file)
                root = tree.getroot()
                for single_object in root.findall('object'):
                    single_object[0].text = single_object[0].text.lower()
                    if "_scrollable" in single_object[0].text:
                        single_object[0].text = "scrollable"
                tree.write(file)
    
print("finished")