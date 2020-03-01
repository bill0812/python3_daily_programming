#from xml.etree.ElementTree import Element, SubElement, tostring
from lxml.etree import Element, SubElement, tostring, ElementTree
from xml.dom.minidom import parseString
from argparse import ArgumentParser

# node_root = Element('annotation')
 
# node_folder = SubElement(node_root, 'folder')
# node_folder.text = 'hand gesture'
 
# node_filename = SubElement(node_root, 'filename')
# node_filename.text = '000001.jpg'
 
# node_size = SubElement(node_root, 'size')
# node_width = SubElement(node_size, 'width')
# node_width.text = '500'
 
# node_height = SubElement(node_size, 'height')
# node_height.text = '375'
 
# node_depth = SubElement(node_size, 'depth')
# node_depth.text = '3'
 
# node_object = SubElement(node_root, 'object')
# node_name = SubElement(node_object, 'name')
# node_name.text = 'mouse'
# node_difficult = SubElement(node_object, 'difficult')
# node_difficult.text = '0'
# node_bndbox = SubElement(node_object, 'bndbox')
# node_xmin = SubElement(node_bndbox, 'xmin')
# node_xmin.text = '99'
# node_ymin = SubElement(node_bndbox, 'ymin')
# node_ymin.text = '358'
# node_xmax = SubElement(node_bndbox, 'xmax')
# node_xmax.text = '135'
# node_ymax = SubElement(node_bndbox, 'ymax')
# node_ymax.text = '375'
 
# # Formatted display, the newline of the newline
# xml = tostring(node_root) 
# dom = parseString(xml)
# dom = dom.toprettyxml(indent='\t')
# # print(type(dom.toprettyxml(indent='\t')))
# # print(type(xml))

# with open("test.xml", "wb") as f :
#     f.write(dom.encode("utf-8

# if __name__ == "__main__":
#     parser = ArgumentParser('Process Image With Histogram Equalization.')
#     parser.add_argument('--image', required = True , metavar='path/to/image', dest="img_name", help="Input your image.")
#     parser.add_argument('--output', metavar='path/to/output/directory', dest="out_dir", help="Output image to a specific directory", default="output")
#     parser.add_argument('--choice', required = True , type=int , choices = [0,1,2,3], dest="choice", help="0 for gray image.\n1 for RGB image.\n2 for HSV image.", default="0")
#     args = parser.parse_args()

import os, fnmatch
from os import walk
import json

# 指定要列出所有檔案的目錄
mypath = "dataset"

# check for specific type of file
pattern = "*.json"  
i = 1
# 遞迴列出所有子目錄與檔案
for root, dirs, files in walk(mypath):
    # print("Current Directory：", root)
    # print("Current Directory's Files：", files)
    for each_one in files: 
        if fnmatch.fnmatch(each_one, pattern):
            if i < 2 :
                filename = os.path.join(root,each_one)
                with open(filename) as f :
                    data = json.load(f)
                print(data["shapes"])
                # print(filename)

                i += 1
