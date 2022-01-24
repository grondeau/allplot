# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 20:35:23 2021

@author: Gary
"""

from bs4 import BeautifulSoup as bs

config_file_name = "allplot.xml"


def defaultXML():
    
    content = ''  # null content
    
    y = bs(content, features="html.parser")
    tag = y.new_tag(name="config")
    tag.string = "Configuration file for ALLPlots program"
    y.insert(1,tag)
    
    ### File Mangement
    
    tag = y.new_tag("files")
    tag.string = "Working or default file names"
    y.config.insert(2,tag)
    
    tag = y.new_tag("filea")
    tag["FileA"]="ALL.txt"
    y.config.files.insert(3,tag)
    
    tag = y.new_tag("descriptiona")
    tag["DescriptionA"]=" "
    y.config.files.insert(4,tag)
    
    tag = y.new_tag("fileb")
    tag["FileB"]="ALL.txt"
    y.config.files.insert(5,tag)
    
    tag = y.new_tag("descriptionb")
    tag["DescriptionB"]=" "
    y.config.files.insert(6,tag)
    
    tag = y.new_tag("SaveMatchSortDescrptA")
    tag["SaveMatchSortDescrptA"]=" "
    y.config.files.insert(8,tag)
    
    tag = y.new_tag("SaveMatchSortDescrptB")
    tag["SaveMatchSortDescrptB"]=" "
    y.config.files.insert(9,tag)
    
    tag = y.new_tag("lastMatchSortDescrptA")
    tag["lastMatchSortDescrptA"]=" "
    y.config.files.insert(10,tag)
    
    tag = y.new_tag("lastMatchSortDescrptB")
    tag["lastMatchSortDescrptB"]=" "
    y.config.files.insert(11,tag)
    
    ### Settings
    
    tag = y.new_tag("settings")
    tag.string = "Program settings"
    y.config.insert(2,tag)
    
    tag = y.new_tag("band")
    tag["band"]="All Bands"
    y.config.settings.insert(2,tag)
    
    tag = y.new_tag("dpi")
    tag["DPI"]=200
    y.config.settings.insert(3,tag)
    
    tag = y.new_tag("xsize")
    tag["xsize"]=5.0
    y.config.settings.insert(5,tag)
    
    tag = y.new_tag("ysize")
    tag["ysize"]=4.0
    y.config.settings.insert(6,tag)
    
    tag = y.new_tag("view")
    tag["view"]="WO"
    y.config.settings.insert(7,tag)
    
    tag = y.new_tag("shadow")
    tag["shadow"]="True"
    y.config.settings.insert(8,tag)
    
    tag = y.new_tag("mapimage")
    tag["mapimage"]="False"
    y.config.settings.insert(9,tag)
    
    tag = y.new_tag("clrscalemx")
    tag["clrscalemx"]=10.0
    y.config.settings.insert(10,tag) 
    
    tag = y.new_tag("clrscalemn")
    tag["clrscalemn"]=-10.0
    y.config.settings.insert(11,tag) 
    
    tag = y.new_tag("scalemx")
    tag["scalemx"]=15.0
    y.config.settings.insert(12,tag) 
    
    tag = y.new_tag("scalemn")
    tag["scalemn"]=-15.0
    y.config.settings.insert(13,tag) 
    
    tag = y.new_tag("pautoscale")
    tag["pautoscale"]="True"
    y.config.settings.insert(14,tag)
    
    tag = y.new_tag("latitude")
    tag["latitude"]=44.0
    y.config.settings.insert(15,tag)
    
    tag = y.new_tag("longitude")
    tag["longitude"]=-123.0
    y.config.settings.insert(16,tag)
    
    ###  Status
    
    tag = y.new_tag("status")
    tag.string = "Program status"
    y.config.insert(3,tag)
    
    tag = y.new_tag("havematches")
    tag["havematches"]="False"
    y.config.status.insert(3,tag)
    
    tag = y.new_tag("havesorted")
    tag["havesorted"]="False"
    y.config.status.insert(4,tag)
    
    tag = y.new_tag("havereference")
    tag["havereference"]="False"
    y.config.status.insert(5,tag)
    
    tag = y.new_tag("inlineplots")
    tag["inlineplots"]="True"
    y.config.status.insert(6,tag)
    
    tag = y.new_tag("savedplots")
    tag["savedplots"]="True"
    y.config.status.insert(6,tag)
    
    
    f = open(config_file_name, "w")
    f.write(y.prettify())
    f.close()
    
    
def updateXML(tag,value):
    try:
        f = open(config_file_name, "r")
        content = f.read()
        f.close()
    except:
        print(config_file_name," file not found")
        return
           
    soup = bs(content, features='html.parser')
    soup.find(tag)[tag] = value 
    
    f = open(config_file_name, "w")
    f.write(soup.prettify())
    f.close()

def getXML(tag):
    try:
        f = open(config_file_name, "r")
        content = f.read()
        f.close()
    except:
        print(config_file_name," file not found")
        return
           
    soup = bs(content, features='html.parser')
    value = soup.find(tag)[tag]
    return value
    
def readXML():
    
    with open(config_file_name) as f:
        content = f.read()
    f.close()
    
    soup = bs(content,features='html.parser')
    
    print(soup)
    
##### Main ####    
    
#defaultXML()    

#readXML()

#updateXML('filea','junk')

#readXML()