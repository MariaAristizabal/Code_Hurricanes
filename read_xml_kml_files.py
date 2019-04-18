#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 07:05:13 2019

@author: aristizabal
"""

#%%

from lxml import etree
import xml.etree.ElementTree as ET

xml_file = '/Users/aristizabal/Desktop/CS_OFFL_SIR_SINI2__20180101T000350_20180101T000445_C001.HDR.xml'

doc_xml = open(xml_file).read()
doc_xml = bytes(bytearray(doc_xml, encoding = 'utf-8'))

xml2df = XML2DataFrame(doc_xml)
xml_dataframe = xml2df.process_data()
    
from bs4 import BeautifulSoup

doc_xml = open(xml_file).read()
soup = BeautifulSoup(doc_xml,'lxml')  

for x in soup.find_all('data_set_name'):
    print(x)
    
from fastkml import kml

kml_file = '/Users/aristizabal/Desktop/CS_OFFL_SIR_GOP_2__20181006T033524_20181006T042503_C001.kml'

doc_kml = open(kml_file).read()
doc_kml = bytes(bytearray(doc, encoding = 'utf-8'))
soup = BeautifulSoup(doc_kml,'lxml')  
k = kml.KML()
k.from_string(doc)