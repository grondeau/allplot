 # -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 23:24:47 2021

@author: Gary
"""
import os
from os.path import exists
import config as xml
import numpy as np
import pandas as pd
import math
from symfit import parameters, variables, sin, cos, Fit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import tkinter as tk
import tkinter.filedialog as fd
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.feature.nightshade import Nightshade
from bs4 import BeautifulSoup as bs

 
Version = '1.0'  
BuildDate = "2022-1-21"
df = pd
df2 = pd
mapimage = True
shadow = True

autoscale = True
havematches = False
havesorted = False
havereference = False
inlineplots = True
savedplots = True
fitdata = False
strng = ' '
extent = [-105, 63, 0,-40]  #world
view = 'WO'
clat = 44.0
clon = -124.0
dpi = 200
fsx = 6.0
fsy= 5.0
nfit = 3



def setExtent(stuff):  
    global extent, view  
    if stuff == 'WO' or stuff == 'wo':
        extent = [ -160+clon,160+clon , 0,-60]
        view = 'WO'
        strng = "World View"
    elif stuff == 'NA' or stuff == 'na':
        extent = [-145, -60, 70,0]
        view = 'NA'
        strng = "North America"
    elif stuff == 'AM' or stuff == 'am':
        view = 'AM'
        extent = [-145, -28, 55,-55]
        strng = "Americas, N & S"
    elif stuff == 'NH' or stuff == 'nh':
        view = 'NH'
        extent = [-200, 63, 90,0] 
        strng = "Northern Hemisphere"
    else:
#        extent = [-105, 63, 0,-40]
        extent = [-180, 180, 0,-65]
        view = 'WO'
    xml.updateXML('view',view) 
    return strng
    
def split_path(path,code):
    head, tail = os.path.split(path)
    if code =='name':
        return tail
    elif code =='path':
        return head
    else:
        print("error roger wilco")
        

try:
    f1 = open('allplot.xml','r') 
    f1.close()
except:
    print("allplot.xml not found")
    xml.defaultXML() 
    print("Generating Default XML file")

    
strng = xml.getXML('havematches')
havematches = ("True" == strng)    

strng = xml.getXML('havesorted')
havesorted = ("True" == strng)

strng = xml.getXML('havereference')
havereference = ("True" == strng)

strng = xml.getXML('inlineplots')
inlineplots = ("True" == strng)

strng = xml.getXML('savedplots')
savedplots = ("True" == strng)

strng = xml.getXML('shadow')
shadow = ("True" == strng)

strng = xml.getXML('mapimage')
mapimage = ("True" == strng)

strng = xml.getXML('pautoscale')
autoscale = ("True" == strng)

strng = xml.getXML('fitdata')
fitdata = ("True" == strng)

strng = xml.getXML('fitorder')
nfit = int(strng)

strng = xml.getXML('latitude')
clat = float(strng)

strng = xml.getXML('longitude')
clon = float(strng)

strng = xml.getXML('dpi')
dpi = int(strng)

strng = xml.getXML('xsize')
fsx = float(strng)

strng = xml.getXML('ysize')
fsy = float(strng)



havemeans = False

antA = 'Default'
antB = 'Default'
fileA = xml.getXML('filea')
fileB = xml.getXML('fileb')
antA = xml.getXML('descriptiona')
antB = xml.getXML('descriptionb')

spath = split_path(fileA,"path")

maxcolor = float(xml.getXML('clrscalemx'))
mincolor = float(xml.getXML('clrscalemn'))

minscale = float(xml.getXML('scalemn'))
maxscale = float(xml.getXML('scalemx'))

MaxSig=40

minsnrab = 0
maxsnrab = 0

first = True

#rightnow = datetime.datetime.now()
#print('Now:',rightnow)


delta_t = 3  #minutes between frames
int_t = 10   #minutes of collected data per frame
imin = 0     #global index for minute iteration
frate = 5

SMALL_SIZE = 6
MEDIUM_SIZE = 9
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
plt.rcParams["figure.dpi"] = dpi   
plt.rcParams["scatter.edgecolors"] = "None"
plt.rcParams["lines.linewidth"] = 0.2
plt.rcParams["figure.autolayout"]=True
#plt.rcParams["animation.ffmpeg_path"]='ffmpeg'

def fourier_series(x, f, n=0):
    """
    Returns a symbolic fourier series of order `n`.

    :param n: Order of the fourier series.
    :param x: Independent variable
    :param f: Frequency of the fourier series
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x) 
                     for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series

x, y = variables('x, y')
#w, = parameters('w')
model_dict = {y: fourier_series(x, f=1, n=nfit)}
print('Fit model: ',model_dict)


def onf(flag):
    if flag:
        return "ON"
    else:
        return "OFF"

def status():
    global bnd,view
    global Version, BuildDate
    print('\nALLPLOT version',Version,'  Built',BuildDate,' by Gary Rondeau, AF7NX \n')
    
    print("MapView:",setExtent(xml.getXML('view')))
    print("\nWorking files:")
    print('A:',antA,' ',fileA)
    print('B:',antB,' ',fileB)
    print("Saved plots PATH: ",spath,'\n')
    
    print("Inline plots are",onf(inlineplots),"  Saved plots are",onf(savedplots))
    if havesorted:
        print('Files already matched and processed')
    elif havematches:
        print('Matches already found for these files.')
        
    bnd = xml.getXML('band')
    if bnd!='All Bands':
        print(bnd,' band selected')
    else:
        print(bnd,' selected')    
        
    if havereference:
        LantA = xml.getXML('lastmatchsortdescrpta') 
        LantB = xml.getXML('lastmatchsortdescrptb') 
        print("\nSorted matchfile for reference antenna,",LantA,", and",LantB,'saved as LastMatchSort.csv')
    else:
        print("\nNo valid reference antenna")


class animationGeneral(object):
# First set up the figure, the axis, and the plot element we want to animate
    def __init__(self, val1,lon1,lat1,val2,lon2,lat2,starttime,nframes,delta, Afname, Bfname, valcol):
        #Global definitions   
        
        
        self.value=val1
        self.lonVar=lon1
        self.latVar=lat1
        self.value2=val2
        self.lonVar2=lon2
        self.latVar2=lat2
        self.t=0
        self.stime=starttime
        self.fnameA=Afname
        self.fnameB=Bfname
        self.vcolm = valcol
        self.frames=nframes
        self.dt=delta
             
        self.start_time = time.time()     
        
        #Figure
        self.plt = plt
        self.fig = plt.figure(dpi=dpi, figsize = (fsx,fsy))
        
        self.firstime = True
        self.twofile = False
        
        # this module will know what to do based upon self.vcolm and self.fnameB
        # for single file multiband animateion valcom="color" and fnameB = 'None'
        # for SNR A-B plots, valcom="SNR_AB", fnameB will be populated
        # for two radio comparisons, valcom="color", fnameB is NOT 'None'
        
        
        
        if self.vcolm  == 'color':    #'SNR_AB' for A-B plots
            self.firstime = False

        #Axes definition
        proj = ccrs.AzimuthalEquidistant(central_latitude=clat, central_longitude=clon)  
        self.ax = plt.axes(projection=proj)
        
        AName = self.fnameA.split('/')
        
        Rname = AName[-1].split('.')
        if self.vcolm == "SNR_AB": 
            aniName = spath+'/'+Rname[0]+'AB'+bnd+'.mp4'
            self.ax.set_title("SNR A-B    "+bnd,fontsize= 9)
        elif self.fnameB=='None':
            self.ax.set_title("Reception Reports",fontsize= 9)
            aniName = spath+'/'+Rname[0]+'.mp4'
        else: 
            self.twofile = True
            self.ax.set_title("Reception Reports",fontsize= 9) 
            aniName = spath+'/'+antA[0:5]+'&'+antB[0:5]+'A&B.mp4'
        
        if os.path.isfile(aniName) :
            print(aniName)
            print("File exists already")
            inp =input("Ovewrite? Y/N ->")
            if inp == 'Y' or inp=='y':
                os.remove(aniName)
            else:
                return
        print('Creating ', aniName)
        
        
        
        if view=='WO':
            self.ax.set_ylim(-13000000, 16500000)       
            self.ax.set_xlim(-17000000, 17000000)
        else:    
            self.ax.set_extent(extent,ccrs.PlateCarree())
        
        self.ax.text( 0.0, 1.1,'A: '+AName[-1]+' - '+antA,
                     bbox={'facecolor': 'white', 'alpha': 0.95, 'linewidth':0.0,'pad':1},
                     transform=self.ax.transAxes, ha="left")
        
        if self.fnameB != 'None':
            BName = self.fnameB.split('/')
            self.ax.text( 0.0, 1.06,'B: '+BName[-1]+' - '+antB,
                     bbox={'facecolor': 'white', 'alpha': 0.95, 'linewidth':0.0,'pad':1},
                     transform=self.ax.transAxes, ha="left")
            
        xf = 0.0
        yf = 0.0
        
        if self.vcolm =='color':
            s0 = self.ax.scatter(xf,yf,c='#b3ff30', s=10, label='160m',edgecolor='black',lw=0.2)
            s1 = self.ax.scatter(xf,yf,c='#e550e5', s=10, label='80m',edgecolor='black',lw=0.2)
            s2 = self.ax.scatter(xf,yf,c='#5959ff', s=10, label='40m',edgecolor='black',lw=0.2)
            s3 = self.ax.scatter(xf,yf,c='#29e3b8', s=10, label='30m',edgecolor='black',lw=0.2)
            s4 = self.ax.scatter(xf,yf,c='#f2ac0c', s=10, label='20m',edgecolor='black',lw=0.2)
            s5 = self.ax.scatter(xf,yf,c='#eaea11', s=10, label='17m',edgecolor='black',lw=0.2)
            s6 = self.ax.scatter(xf,yf,c='#cca166', s=10, label='15m',edgecolor='black',lw=0.2)                   
            s7 = self.ax.scatter(xf,yf,c='#b22222', s=10, label='12m',edgecolor='black',lw=0.2)
            s8 = self.ax.scatter(xf,yf,c='#ff69b4', s=10, label='10m',edgecolor='black',lw=0.2)
            s9 = self.ax.scatter(xf,yf,c='grey', s=12, label='A',edgecolor='black',lw=0.2)
            s10 = self.ax.scatter(xf,yf,c='grey', s=6, marker = 's', label='B',edgecolor='black',lw=0.2)
       #     s11 = self.ax.scatter(0,0,c='#831cba', s=15, label='Home',edgecolor='black',lw=0.2)
        

        #The plot to animate
        if self.frames == 0:
            self.frames = 1     #ensure at least one pass throught the update function.
        
        if shadow:
            self.shade = self.ax.add_feature(Nightshade(self.stime, alpha=0.2)) 
            
        self.scat2 = self.ax.scatter(0, 0, c=0, s=1
             , lw=0.5
             ,transform=ccrs.PlateCarree(),edgecolor='none')
        
        if self.twofile:
            self.scat3 = self.ax.scatter(0, 0, c=0, s=1
                 , lw=0.5
                 ,transform=ccrs.PlateCarree(),edgecolor='none')
        
        self.ax.add_feature(cfeature.OCEAN)
        self.ax.add_feature(cfeature.LAND, color='lightgrey')
        self.ax.add_feature(cfeature.BORDERS, linewidth=0.2, linestyle=':')
        self.ax.coastlines(linewidth=0.3)
        if mapimage:
            self.ax.stock_img()
               
        self.ax.gridlines(linewidth=.2,linestyle="dashed",color='black',xlocs=[-150,-120,-90,-60,-30,0,30,60,90,120,150,180], ylocs=[-75,-60,-45,-30,-15,0,15,30,45,60,75])
        self.ax.gridlines(linewidth=.3, linestyle="dashed", ylocs=[0], xlocs=[], color='black')
        
        if self.vcolm =='color' and self.twofile:
            self.ax.legend(handles=[s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10], title='Bands',loc ='upper left', fontsize='small',framealpha=0.9)
        elif self.vcolm =='color':
            self.ax.legend(handles=[s0, s1, s2, s3, s4, s5, s6, s7, s8], title='Bands',loc ='upper left', fontsize='small',framealpha=0.9)
        
#        self.plt.colorbar(pad = .02, location = 'bottom', shrink=0.4, aspect = 40, label=description)
        #Animation definitio
        anim = animation.FuncAnimation(self.fig, self.update, frames=self.frames, interval=1, blit=False)     

     #   FFwriter = animation.FFMpegWriter(fps = frate, extra_args=['-vcodec', 'libx264'])
        anim.save(aniName, writer='ffmpeg', fps = frate)
        self.plt.clf()
    
    def update(self,i):
        global antA, antB
        global bnd
        
        t=i
        
        print('frame', t, 'of', self.frames - 1 ,'T:','{:.2f}'.format(time.time()-self.start_time))
        self.start_time = time.time()
        
        start= self.stime + datetime.timedelta(minutes=i*self.dt)
        end = start + datetime.timedelta(minutes=10)
       
        self.scat2.remove()
        
        if self.twofile:
            self.scat3.remove()
        
        if shadow:
            self.shade.remove()  
        
       
        self.ax.text(1.0, 1.1, start.strftime("%Y-%m-%d  %H:%M UTC"),
            bbox={'facecolor': 'white', 'alpha': 0.8, 'linewidth':0.0, 'pad':1},
            transform=self.ax.transAxes, ha="right")
        
        if shadow:
            self.shade = self.ax.add_feature(Nightshade(start, alpha=0.2))  
                    
        self.scat2 = self.plt.scatter(self.lonVar[start:end], self.latVar[start:end],
                          c=self.value[start:end], cmap='jet', lw=0.2, s=9, alpha=0.8,
                          vmin=mincolor, vmax=maxcolor,
                          transform=ccrs.PlateCarree(),edgecolor='black')
        if self.twofile:
            self.scat3 = self.plt.scatter(self.lonVar2[start:end], self.latVar2[start:end],
                              c=self.value2[start:end], cmap='jet', lw=0.2, s=4, alpha=0.8,
                              vmin=mincolor, vmax=maxcolor,marker = 's',
                              transform=ccrs.PlateCarree(),edgecolor='black')
        
        if self.firstime:
            description = antB+'    $\Delta SNR$ (dB)    '+antA+'  '+bnd   
            self.plt.colorbar(pad = .02, location = 'bottom', shrink=0.4, aspect = 40, label=description)
            self.firstime = False
        
        if self.twofile:
            return self.scat2, self.scat3,
        else:
            return self.scat2,
    
def fakedf():
    fake = ['Lat','Lon','SNR_AB','color']
    df = pd.DataFrame(columns= fake)
    df.loc[0] = [0,0,0,'white']
    return df
    

def mapum():
    global fileA, fileB
    
    f1 = open(fileA,'r')
                
    header = ['TimeStamp', 'Freq', 'Rx/Tx', 'Mode', 'SNR', 'DT', 'AF', 'Target', 'Call', 'Grid', 'color', 'Extra2']
    df = pd.read_csv(fileA, delimiter=r"\s+", names=header,dtype={'color': 'str', 'Extra2': 'str'})  # read in the entire ALL.txt file    
    f1.close()
    
    df = df.drop(columns=['Rx/Tx', 'Mode', 'DT', 'AF', 'Target','Extra2'])
    
    if fileB == "ALL.txt":
        fileB = 'None'
        df2 = fakedf()
        
    else:
        f2 = open(fileB,'r')
        df2 = pd.read_csv(fileB, delimiter=r"\s+", names=header,dtype={'color': 'str', 'Extra2': 'str'})  # read in the entire ALL.txt file    
        f2.close()
        
        print("A and B files will be animated. Fixing up B...")
        df2 = df2.drop(columns=['Rx/Tx', 'Mode', 'DT', 'AF', 'Target','Extra2'])
        
        df2['Grid']= df2['Grid'].str.extract(r'^([A-R][A-Q][0-9]{2})', expand=False)
        df2['Grid'] = df2.groupby('Call')['Grid'].fillna(method = 'ffill')
        df2['Grid'] = df2.groupby('Call')['Grid'].fillna(method = 'bfill')
        
        df2 = df2.sort_values(['Grid','Call'],ascending=False)
        df2 = df2.set_index('Call')
        
        try:
            f4 = open("Grids.csv",'r')
            grds = pd.read_csv("Grids.csv")
            f4.close()  
            print('Repair missing grids with Grids.csv file data... ')
            grds = grds.set_index('Call')       
            df2['Grid'].update(grds.Grid)
        
        except:
            print("Grids.csv file not found.  No attempt to rescue lost grids...")
        
        df2=df2.reset_index(drop=False)
        df2 = df2.sort_values(['Grid','Call'],ascending=False)
        
        df2['Lon'] = df2['Grid'].apply(Lonfrmgrid)
        df2['Lat'] = df2['Grid'].apply(Latfrmgrid)
        df2['band'], df2['color'] = zip(*df2['Freq'].apply(getband)) 
        print("FileB:",len(df2),"reports")
           
       
    df['Grid']= df['Grid'].str.extract(r'^([A-R][A-Q][0-9]{2})', expand=False)
    
    print('Fixing up A file grids...')
    
    df['Grid'] = df.groupby('Call')['Grid'].fillna(method = 'ffill')
    df['Grid'] = df.groupby('Call')['Grid'].fillna(method = 'bfill')
    
    df = df.sort_values(['Grid','Call'],ascending=False)
    df = df.set_index('Call')
    
    try:
        f4 = open("Grids.csv",'r')
        grds = pd.read_csv("Grids.csv")
        f4.close()  
        print('Repair missing grids with Grids.csv file data... ')
        grds = grds.set_index('Call')       
        df['Grid'].update(grds.Grid)
    
    except:
        print("Grids.csv file not found.  No attempt to rescue lost grids...")
    
    df=df.reset_index(drop=False)

#    print(df) 

    df = df.sort_values(['Grid','Call'],ascending=False)        
    df['Lon'] = df['Grid'].apply(Lonfrmgrid)
    df['Lat'] = df['Grid'].apply(Latfrmgrid)
    print('setting bands and colors...')
 
    df['band'], df['color'] = zip(*df['Freq'].apply(getband))  
    print("FileA:",len(df),"reports")
    #print(df)   
    
    animum(df,df2,'color')
      
    
def animum(df,df2,col):    
    twofiles = False
    if col=='color' and fileB != 'None':
        twofiles = True
    #Turn TimeStamp into datetime value
   
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'], format='%y%m%d_%H%M%S')
    df.reset_index(drop=True, inplace=True)
    
    if twofiles:
        df2['TimeStamp'] = pd.to_datetime(df2['TimeStamp'], format='%y%m%d_%H%M%S')
    #Remove things not Grids from the Grids column
   
   
    
    starttime1 = df.TimeStamp[0]
    lastone = len(df)
    endtime1 = df.TimeStamp[lastone-1]
   # print(df.TimeStamp[lastone-1])


    if twofiles:
        starttime2 = df2.TimeStamp[0]
        lastone = len(df2)
        endtime2 = df2.TimeStamp[lastone-1]     
        starttime1 = min(starttime1,starttime2)
        endtime1 = max(endtime1, endtime2)
    
        
    c = endtime1 - starttime1
    minutes = int(c.total_seconds() / 60)
    nframes = int(minutes/delta_t)
 
    print('Start: ', starttime1)
    print('End: ',endtime1)
    print('nframes', nframes)
    print('delta_t', delta_t)
    
    df = df.sort_values('TimeStamp')
    df.set_index('TimeStamp', inplace=True)
    
    if twofiles:
        df2= df2.sort_values('TimeStamp')
        df2.set_index('TimeStamp', inplace=True)
        
    #print(df)

    animationGeneral(df[col],df.Lon, df.Lat, df2[col], df2.Lon, df2.Lat, starttime1, nframes ,delta_t, fileA, fileB, col)       
    #a.show()
    return

    
def mapAB():
    if not havesorted:
        print('Process matches first -> PO')
        return
    else:    
        try: 
            f1 = open('MatchSort.csv','r')
        except:
            print('MatchSort.csv file not found')
            return
    df = pd.read_csv('MatchSort.csv')  
    f1.close()   
    df2 = fakedf()
    
    df = df.drop(columns=['index','Call','MaxA','MaxB','Color','SNR_A','SNR_B'])  
 #   print('MatchSort after dropping junk')
 #   print(df)    
    grouped = df.groupby(['TimeStamp','Grid','Band'])  
 #   print("grouped by Time and grid in prep for averaging")
 #   print(grouped)   
    gridavs = grouped.mean() 
    gridavs = gridavs.sort_values(["TimeStamp"]).reset_index()
 #   print("averaged over grouping we hope")
  #  print(gridavs)
    
    #gridavs = gridavs.sort_values(["TimeStamp"]).reset_index()
    #gridavs['ma']=gridavs.rolling(window=int_t).mean()
    #print("compute rolling averages")
    #print(gridavs)
    
    if bnd in gridavs.Band.values:    
        gridavs=gridavs[gridavs.Band == bnd] 
        print(bnd,' band only')
    else:
        print('Using all bands')
    
#    print("to go to animum")
#    print(gridavs)    
             
    
    animum(gridavs,df2,'SNR_AB')
       
#    matches.drop()
    

def getband(freq):   #returns a lable and a color
    if freq >= 1.8 and freq<= 2.0:        
        b= "160m"
        c= '#b3ff30'  # '#7cfc00'
    elif freq >= 3.5 and freq <= 4.0:
        b= "80m"
        c= '#e550e5' 
    elif freq >=7.0 and freq <= 7.4:
        b= "40m"
        c= '#5959ff'
    elif freq >=10.1 and freq <= 10.15:
        b= "30m"
        c= '#29e3b8'   # #62d962'
    elif freq >=14.0 and freq <= 14.4:
        b= "20m"
        c= '#f2ac0c'
    elif freq >=18.07 and freq <= 18.17:
        b= "17m"
        c= '#eaea11'
    elif freq >=21.0 and freq <= 21.4:
        b= "15m"
        c= '#cca166' 
    elif freq >=24.89 and freq <= 24.99:
        b= "12m"
        c= '#b22222'
    elif freq >=28.0 and freq <= 29.0:
        b= "10m"
        c= '#ff69b4'
    else:
        b= "  ",
        c= '' 
        
    return b,c
    
def cleangrid(grid):
    if len(grid) != 4:
        return ('')
    if grid[0] < 'A' or grid[0] > 'R':
        return ('')
    if grid[1] < 'B' or grid[1] > 'Q':
        return ('')
    if grid[2] < '0' or grid[2] > '9':
        return ('')
    if grid[3] < '0' or grid[3] > '9':
        return ('')
    
    
    return(grid) 

def isNaN(string):
    return string != string
        
def Lonfrmgrid(grid):   
    if isNaN(grid):
        return
    a = 20*(ord(grid[0])-65) + 2*(ord(grid[2]) - ord('0') +0.5) - 180         
    return (float(a))

def Latfrmgrid(grid):  
    if isNaN(grid):
        return  
    c = 10*(ord(grid[1])-65) + ord(grid[3]) - ord('0') +0.5 - 90    
    return (float(c))
    
def matches():    #Called by CF
    global aline_field, bline_field
    global fileA, fileB
    global havematches
    
    pointer = 0
    rwpointer = 0
    cts = ' '
    try: 
        f1 = open(fileA,'r')
    except:
        print(fileA, 'not found')
        return
    try:
        f2 = open(fileB,'r')
    except:
        print(fileB, ' not found')
        return
    
    f3 = open('Matches.csv',"w")       #working file    
    header =  'MaxA MaxB TimeStamp Band Color SNR_A SNR_B Call Grid\n'
    f3.write(header)

    print("Processing Input Files Looking for Matches") 
    counter = 0    
    while True:
        aline = f1.readline()
        if aline=='':
            break
        aline = aline.replace('<','')
        aline = aline.replace('>','')         
        aline_field = aline.split()   
        f2.seek(rwpointer)
#              0        1    2      3    4    5     6       7      8    9    10    11    
#ALL.TX::  Timestamp  Freq Rx/Tx  Mode  SNR  DT  AudioF  Target  Call Grid More1 More2
    
        if not cts == aline_field[0]:   #if new aline has same timestamp, rewind b file
            cts = aline_field[0]
            amxreport = int(aline_field[4])   #initialize mx report for this time stamp
            bmxreport = -99
            first = True
        else:
            if int(aline_field[4]) > amxreport:
                amxreport = int(aline_field[4])
        while True :
            pointer = f2.tell()
            bline = f2.readline()
            if bline=='':
                break
            bline = bline.replace('<','')
            bline = bline.replace('>','')  
            bline_field = bline.split()
            if cts > bline_field[0]:
                continue
            if int(bline_field[4]) > bmxreport:
                bmxreport = int(bline_field[4])
            if first and cts==bline_field[0]:
                first = False
                rwpointer = pointer
            if cts == bline_field[0]:
                if bline_field.__len__() >= 10 and aline_field.__len__() >= 10:
                    if bline_field[8] == aline_field[8]:
                        if len(bline_field[8])<=3:
                            if (bline_field[8].isalpha() or bline_field[8].isdigit()) and bline_field[9]==aline_field[9]:
                                aline_field[8] = aline_field[9]
                                try:
                                    aline_field[9] = aline_field[10]
 #                                   print("fixed:", aline_field[8], aline_field[9])
                                except:
                                    print('No Call:',bline_field[8],aline_field[9])

                        matchstring = str(amxreport) + ' '+ str(bmxreport)+ ' ' + aline_field[0] +' ' + getband(float(aline_field[1]))[0] + ' ' + getband(float(aline_field[1]))[1] +' '+ aline_field[4]+' '+bline_field[4]+' '+aline_field[8]+' '+cleangrid(aline_field[9])+"\n"
#                       print (matchstring)
                        f3.writelines(matchstring)  
                        counter += 1   
            if cts  < bline_field[0]:
                break
    print(counter,' Matches found...')
    havematches = True
    xml.updateXML('havematches','True')
    
    f1.close()
    f2.close()
    f3.close()  
     
     
def Distance(grid):
    a=[ord(x) for x in grid]
    lon = 20*(a[0]-65) + 2*(a[2] - ord('0')) - 180
    lat = 10*(a[1]-65) + a[3] - ord('0') -90  
    R = 3958.8  #radius of the Earth miles
        
    
    lat1 = math.radians(clat)
    lon1 = math.radians(clon)
    lat2 = math.radians(lat)
    lon2 = math.radians(lon)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c

    return(distance)

def Bearing(grid):  
    a=[ord(x) for x in grid]
    lon = 20*(a[0]-65) + 2*(a[2] - ord('0')) - 180
    lat = 10*(a[1]-65) + a[3] - ord('0') -90
   
    lat1 = math.radians(clat)
    lat2 = math.radians(lat)
    diffLong = math.radians(lon - clon)

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # Now we have the initial bearing but math.atan2 return values
    # from -180° to + 180° which is not what we want for a compass bearing
    # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360    
    return(compass_bearing)

def makelegend(ax):
    #ax = fig.add_subplot(projection='polar',facecolor="#f6f6f6")
    s0 = ax.scatter(0,0,c='#b3ff30', s=10, label='160m',edgecolor='black',lw=0.2)
    s1 = ax.scatter(0,0,c='#e550e5', s=10, label='80m',edgecolor='black',lw=0.2)
    s2 = ax.scatter(0,0,c='#5959ff', s=10, label='40m',edgecolor='black',lw=0.2)
    s3 = ax.scatter(0,0,c='#29e3b8', s=10, label='30m',edgecolor='black',lw=0.2)
    s4 = ax.scatter(0,0,c='#f2ac0c', s=10, label='20m',edgecolor='black',lw=0.2)
    s5 = ax.scatter(0,0,c='#eaea11', s=10, label='17m',edgecolor='black',lw=0.2)
    s6 = ax.scatter(0,0,c='#cca166', s=10, label='15m',edgecolor='black',lw=0.2)                   
    s7 = ax.scatter(0,0,c='#b22222', s=10, label='12m',edgecolor='black',lw=0.2)
    s8 = ax.scatter(0,0,c='#ff69b4', s=10, label='10m',edgecolor='black',lw=0.2)
    ax.legend(handles=[s0, s1, s2, s3, s4, s5, s6, s7, s8], title='Bands',loc =[-0.1,0.0], fontsize='small',framealpha=0.9)

def bandcolor(band):
    if band == "160m":                
        c= '#b3ff30'  # '#7cfc00'
    elif band == "80m":     
        c= '#e550e5' 
    elif band == "40m":
        c= '#5959ff'
    elif band == "30m":
        c= '#29e3b8'   # #62d962'
    elif band == "20m":
        c= '#f2ac0c'
    elif band == "17m":
        c= '#eaea11'
    elif band == "15m":
        c= '#cca166' 
    elif band == "12m":
        c= '#b22222'
    elif band == "10m":
        c= '#ff69b4'
    else:
        c= '' 
    return c

def process():      #Called by PO
    global bnd, dpi
    global antA, antB
    global MaxSig
    global minsnrab, maxsnrab
    global havemeans, havematches, havesorted, havereference, autoscale
    
    if not havematches:
        print('Find the matches first -> CF')
        return
    
    if not havesorted:
        #File operations closed - ready to read Matches into a dataframe
        f3 = open("Matches.csv",'r')            
        df = pd.read_csv("Matches.csv",sep=' ')
        f3.close()
        print('Matches.csv read into dataframe...')
        
        nmatches = len(df)
    
        df['Grid'] = df.groupby('Call')['Grid'].fillna(method = 'ffill')
        df['Grid'] = df.groupby('Call')['Grid'].fillna(method = 'bfill')
    
        df = df.sort_values(['Grid','Call'],ascending=False)
        df = df.set_index('Call')
        
        try:
            f4 = open("Grids.csv",'r')
            grds = pd.read_csv("Grids.csv")
            f4.close()  
            print('Repair missing grids with Grids.csv file data... ')
            grds = grds.set_index('Call')       
            df['Grid'].update(grds.Grid)
        
        except:
            print("Grids.csv file not found.  No attempt to rescue lost grids...")
        
        df=df.reset_index(drop=False)
    
    #    print(df) 
    
        df = df.sort_values(['Grid','Call'],ascending=False)       
        
        bad_grids = df[df['Grid'].isna()]
        bad_grids = bad_grids.drop(columns=['MaxA','MaxB','TimeStamp','Band','Color','SNR_A','SNR_B'])
        bad_grids.drop_duplicates(inplace=True)
        bad_grids = bad_grids.sort_values(['Call']).reset_index()
    #    bad_grids = bad_grids.drop(columns=['index'])
        
    
        print(bad_grids.shape[0],"Gridless Matches ")
    
    #    print (bad_grids)
        
        bad_grids.to_csv("BadGrids.csv")
    
        print("The Bad Grids can be found in the temporary file BadGrids.csv")
        print("Manually finding the grids and placing Grid and Call in a Grids.csv file")
        print('will resurrect these data points on subsequent runs.')
    
       
        dd = df.groupby('TimeStamp')['MaxA'].max().to_dict()
        df['MaxA'] = df['TimeStamp'].map(dd)
        dd = df.groupby('TimeStamp')['MaxB'].max().to_dict()
        df['MaxB'] = df['TimeStamp'].map(dd)
    
    
        df['Grid'] = df.groupby('Call')['Grid'].fillna(method = 'ffill')
        df['Grid'] = df.groupby('Call')['Grid'].fillna(method = 'bfill')
    
        df = df.assign(OvrLdA=(df['SNR_A']>MaxSig))
    
        df = df.assign(OvrLdB=(df['SNR_B']>MaxSig))
    
        df = df.assign(SNR_AB=(df['SNR_A']-df['SNR_B']))
    
        df.drop(df[df['OvrLdA']==True].index, inplace=True)
        df.drop(df[df['OvrLdB']==True].index, inplace=True)
        df.drop(df[isNaN(df['Grid']) ].index, inplace=True)
        
        df['Lon'] = df['Grid'].apply(Lonfrmgrid)
        df['Lat'] = df['Grid'].apply(Latfrmgrid)
    
        df = df.drop(columns=['OvrLdA','OvrLdB'])
        
        df = df.sort_values(["TimeStamp"]).reset_index()
        
        print('From original',nmatches,', Saving ',len(df),' remaining matches')  
        
        if exists("MatchSort.csv") and not havereference:
        
            try:
                os.rename('MatchSort.csv', 'LastMatchSort.csv')
            except WindowsError:
                os.remove('LastMatchSort.csv')
                os.rename('MatchSort.csv', 'LastMatchSort.csv')  
            
            LantA = xml.getXML('savematchsortdescrpta') 
            LantB = xml.getXML('savematchsortdescrptb') 
            xml.updateXML('lastmatchsortdescrpta',LantA) 
            xml.updateXML('lastmatchsortdescrptb',LantB)  
            print("Sorted matchfile for antennas ",LantA," and ",LantB,' saved as LastMatchSort.csv')
            havereference = True
            xml.updateXML('havereference',True)
            
        
        df.to_csv("MatchSort.csv", index=False)
        xml.updateXML('savematchsortdescrpta',antA)
        xml.updateXML('savematchsortdescrptb',antB)
        havesorted = True
        xml.updateXML('havesorted',True)
    else:
        f3 = open("MatchSort.csv",'r')            
        df = pd.read_csv("MatchSort.csv")
        f3.close()


    print(len(df),'Total Valid  Matches')

    if bnd in df.Band.values:    
        df=df[df.Band == bnd] 
        print(bnd,' band only')
    else:
        print('Data for '+bnd+' not found. Using all bands')
        bnd = 'All bands'
        
 #   print('Process1 - df:', df.columns)
 
    grouped = df.groupby(['Band','Grid','Color'])

    mean = grouped.mean() 
    mean = mean.assign(StdAB=grouped['SNR_AB'].std())
    mean = mean.drop(columns=['MaxA','MaxB','index'])
#    print("Process2 - mean:",mean.columns)

    mean['StdAB'] = mean['StdAB'].fillna(0)
    
    mean = mean.reset_index()
    
 #   print("Process3 -mean after resetindex:",mean.columns)

    mean['Bearing'] = mean['Grid'].apply(Bearing)
    mean['Distance'] =mean['Grid'].apply(Distance)
    
    print('Averages computed, Grids -> Lon, Lat, Bearing, Distance')

    mean = mean[mean.Distance>100]

    mean['rad'] = mean.apply(lambda row: row.Bearing*math.pi/180, axis=1)

    mean.to_csv("Means.csv")
    
    havemeans = True
    #print("Process3:",mean.columns)
    maxsnrab = max(mean.SNR_AB)
    minsnrab = min(mean.SNR_AB)

    fig = plt.figure(num=1, dpi=dpi, figsize = (fsx,fsy))  # 1200 x 1000 pixels
    ax = fig.add_subplot(projection='polar',facecolor="#f6f6f6")
    ax.errorbar(mean['rad'], mean['SNR_AB'], yerr=mean['StdAB'], capsize=0, elinewidth=0.5, linewidth=0, ecolor=mean['Color'],zorder=2)
    ax.scatter(mean['rad'], mean['SNR_AB'], s=8, c=mean['Color'], edgecolors=['#0f0f0f'], linewidths=0.3, cmap='hsv', alpha=0.85,zorder=3)
   
    
   
    
    if fitdata:    
        fit = Fit(model_dict, x=mean.rad, y=mean.SNR_AB)
        fit_result = fit.execute()
        print(fit_result)
        theta  = np.linspace(0.0,2*np.pi,500)
        ax.plot(theta,fit.model(x=theta, **fit_result.params).y, color='green',linewidth=1.0,zorder=4)
      
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rgrids(radii=[-20,-15,-10,-5,0,5,10,15,20], angle=260., fontsize=5)   
    ax.set_rorigin(-50)
    if not autoscale:
        ax.set_rlim(minscale, maxscale)
        
        
    gridlines = ax.yaxis.get_gridlines()
    gridlines[4].set_color("r")
    gridlines[4].set_linewidth(1.0)
    
    AName = fileA.split('/')
    Bname = fileB.split('/')
    
    plt.figtext( 0.05, 0.92, 'A: '+AName[-1], fontsize=7)   
    plt.figtext( 0.05, 0.89, 'B: '+Bname[-1], fontsize=7)
    plt.xlabel('$\Delta SNR$: '+antA+'   -   '+antB+'    '+bnd)
    
    if bnd == "All bands":
        makelegend(ax)
    
    spath = split_path(fileA,"path")

    filename = spath+'/'+antA+'&'+antB+bnd+'Polar.png'
    ax.set_title("SNR difference A-B (dB) vs Compass Bearing", va='bottom',fontsize= 10)
    
    if savedplots:
        plt.savefig(filename)
        print('Saved plot:',filename)
    if inlineplots:
        plt.show()
    else:
        plt.clf()  
    
    pd.options.display.float_format = "{:,.2f}".format
 
    mean = mean.drop(columns=['Color','Lat', 'Lon','rad'])

    print(mean.shape[0],' Unique Band/Grid values to plot')

    byband = mean.groupby('Band').mean()
    byband = byband.drop(columns=['SNR_A','SNR_B','Bearing','Distance'])

    print(' \n SNR differences by Band \n')

    print(byband)

    #write a couple of datafiles ...

    mean.to_csv("mean.csv")
    df.to_csv("stuff.csv")


def ABRef():           #called by PR
    global antA, antB
    global bnd, dpi
    
    #need to have Matchsort File and Lastmatchsot file, command antA descriptors and dirent antB descriptors.
    
    #need to verify this routine is averaging over grid squares...  Why so many points on PR graph
    
    if not havereference:
        print('Two pairs of ALL.TXT files need to be processed first')
        print('The FileAs (FA) are the reference antenna')
        print('The FileBa (FB) are the desired comparison')
        print('Load each pair, find matches (CF), and grid averages (PO)')
        print('Then try this (AR) again')
        return

    LantA = xml.getXML('lastmatchsortdescrpta')
    LantB = xml.getXML('lastmatchsortdescrptb')
    
    if not antA==LantA:
        print('File A descriptions must be identical for the ''A'' reference files')
        print('Found -  last-pair file A:', LantA, '   Current file A:', antA)
        return    
    
    print('Reference FileA descriptor:',antA)
    print('Comparison between:')
    print(antB)
    print (' and')
    print(LantB)
        
    try: 
        f1 = open('LastMatchSort.csv','r')
    except:
        print('LastMatchSort.csv file not found')
        return
    df2 = pd.read_csv('lastMatchSort.csv')  
    f1.close() 
    df2 = df2.rename(columns={"SNR_AB":"SNR_AC"},errors="raise")
    
    try: 
        f2 = open('MatchSort.csv','r')
    except:
        print('MatchSort.csv file not found')
        return
    df = pd.read_csv('MatchSort.csv')  
    f2.close()       
     
#    print('ABRef1 - df:', df.columns)
    

    grouped = df.groupby(['Band','Grid','Color'])
    grouped2 = df2.groupby(['Band','Grid','Color'])
    
    
    
    mean = grouped.mean()
    mean2 = grouped2.mean()
    
    
    mean = mean.assign(StdAB=grouped['SNR_AB'].std())
    mean2 = mean2.assign(StdAC=grouped2['SNR_AC'].std())
    mean = mean.drop(columns=['SNR_A','SNR_B','MaxA','MaxB','index'])
    mean2 = mean2.drop(columns=['SNR_A','SNR_B','MaxA','MaxB','Lon','Lat','index'])
    
    mean = mean.reset_index()
#    print(mean)
    mean2 = mean2.reset_index()
#    mean2 = mean2.drop(columns=['Color','Band'])
#    print(mean2)
    
 #   print("ABRef2 - mean:",mean.columns)
    
    allmean = mean.merge(mean2, how="inner")
    
#    print(allmean)
      
    allmean.drop(allmean[isNaN(allmean['SNR_AC']) ].index, inplace=False)
    allmean.drop(allmean[isNaN(allmean['SNR_AB']) ].index, inplace=False)  
    
    allmean['SNR_BC'] = allmean['SNR_AC'].sub(allmean['SNR_AB'])
    allmean['StdBC'] = (allmean.StdAC*allmean.StdAC + allmean.StdAB*allmean.StdAB)**1/2
    
    allmean['StdBC'] = allmean['StdBC'].fillna(0)
    
    allmean = allmean.reset_index()
    
    
    allmean['Bearing'] = allmean['Grid'].apply(Bearing)
    allmean['Distance'] =allmean['Grid'].apply(Distance)
    
    allmean = allmean[allmean.Distance>100]
    allmean['rad'] = allmean.apply(lambda row: row.Bearing*math.pi/180, axis=1)
    
    allmean.to_csv("Allmeans.csv")
    
    if bnd in allmean.Band.values:    
         allmean=allmean[allmean.Band == bnd] 
         print(bnd,' band only')
    else:
         print('Using all bands')
    
 #   print(allmean.columns)
 #   print(allmean)
     
    fig = plt.figure(figsize =(fsx,fsy), dpi=dpi)
    ax = fig.add_subplot(projection='polar',facecolor="#f6f6f6")
    ax.errorbar(allmean['rad'], allmean['SNR_BC'], yerr=allmean['StdBC'], capsize=0, elinewidth=0.3, linewidth=0, ecolor=allmean['Color'],zorder=2)
    ax.scatter(allmean['rad'], allmean['SNR_BC'], s=8, c=allmean['Color'], edgecolors=['#0f0f0f'], linewidths=0.3, cmap='hsv', alpha=0.75,zorder=3)
    
    if fitdata:    
        fit = Fit(model_dict, x=allmean.rad, y=allmean.SNR_BC)
        fit_result = fit.execute()
        print(fit_result)
        theta  = np.linspace(0.0,2*np.pi,500)
        ax.plot(theta,fit.model(x=theta, **fit_result.params).y, color='green',linewidth=1.0,zorder=4)
        
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rgrids(radii=[-20,-15,-10,-5,0,5,10,15,20], angle=260., fontsize=4)   
    ax.set_rorigin(-40)
    ax.set_rlim(-20, 20)
    gridlines = ax.yaxis.get_gridlines()
    gridlines[4].set_color("r")
    gridlines[4].set_linewidth(1.0)   
    
    if bnd == "All bands":
        makelegend(ax)
    
    ref = xml.getXML('lastmatchsortdescrpta')
    Bdes = xml.getXML('descriptionb')
    Cdes = xml.getXML('lastmatchsortdescrptb')
    
    plt.figtext( .2, 0.92, 'B: '+Bdes, fontsize=5)   
    plt.figtext( .2, 0.89, 'C: '+Cdes, fontsize=5)
    plt.figtext( .2, 0.86, 'Ref: '+ref, fontsize=5)
    plt.xlabel('$\Delta SNR$: '+Bdes+'   -   '+Cdes+'    '+bnd)


    ax.set_title("SNR difference B-C (dB) from A reference", va='bottom',fontsize= 9)
    
    spath = split_path(fileA,"path")
    filename = spath+'/'+LantB+'&'+antB+bnd+'Polar.png'
    
    if savedplots:
        plt.savefig(filename)
        print('Saved plot:',filename)
    if inlineplots:
        plt.show()
    else:
        plt.clf()
    
        
  
def mapsigs(code):
    global antA, antB
    global bnd, dpi
    global extent, view
    global clat, clon
    global mapimage
    
    AName = fileA.split('/')
    Bname = fileB.split('/')
    
    if code=='AB' and not havemeans:
        print('PO the data for the new band selection first')
        return
        
    fig = plt.figure(figsize =(fsx,fsy), dpi=dpi)
      
    proj = ccrs.AzimuthalEquidistant(central_latitude=clat, central_longitude=clon)
    ax = fig.add_subplot(1,1,1,projection=proj)
#    ax = plt.axes(projection=proj)
    ax.add_feature(cfeature.OCEAN)    
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, linestyle=':')
    ax.coastlines(linewidth=0.3)
    
    
    print('Building Map...')
    
#   if shadow:      
#        ax.add_feature(Nightshade(date, alpha=0.4))
            
    if mapimage:
        ax.stock_img()
        print('Land Image added...') 
    else: 
        ax.add_feature(cfeature.LAND, color='lightgrey')    
        
    ax.gridlines(linewidth=.2,linestyle="dashed",color='black',xlocs=[-150,-120,-90,-60,-30,0,30,60,90,120,150,180], ylocs=[-75,-60,-45,-30,-15,0,15,30,45,60,75])
    ax.gridlines(linewidth=.3, linestyle="dashed", ylocs=[0], xlocs=[], color='black')
    ax.set_extent(extent,crs=ccrs.PlateCarree())  
    if view=='WO':
        ax.set_ylim(-13000000, 16500000)
    
    #print('Ylim',ax.get_ylim())
    
    spath = split_path(fileA,"path")   
    if code == 'AB':
        ax.set_title("SNR difference A-B  "+bnd,fontsize= 9)
        filename = spath+'/'+antA[0:4]+'&'+antB[0:4]+bnd+view+'.png'
    elif code == 'BC':
        ax.set_title("SNR difference B-C  "+bnd,fontsize= 9)
        LantB = xml.getXML('lastmatchsortdescrptb')
        filename = spath+'/'+LantB[0:5]+'&'+antB[0:5]+bnd+view+'.png'
    if code == 'AB':
        f3 = open("Means.csv",'r')
        df = pd.read_csv("Means.csv",sep=',')
        data = 'SNR_AB'
    elif code == 'BC': 
        if not havereference:
            print('Two pairs of ALL.TXT files need to be processed first')
            print('The FileAs (FA) are the reference antenna')
            print('The FileBa (FB) are the desired comparison')
            print('Load each pair, find matches (CF), and grid averages (PO)')
            print('Before the polar comparison will work, use AR')
            print('Then try this (PR) again')
            return
        try:
            f3 = open("Allmeans.csv",'r')
            df = pd.read_csv("Allmeans.csv",sep=',')
        except:
            print('Allmeans.csv not found')
            return
        data = 'SNR_BC'
        #print(df['SNR_BC'])    
    if bnd in df.Band.values:    
        df=df[df.Band == bnd] 
        print(bnd,' band only')
    else:
        print('Using all bands')
        
    if view=='WO' or view=='NH':
        dotsize= 22
    elif view=='AM':
        dotsize = 30
    elif view =='NA':
        dotsize = 40
    
    plt.scatter(df.Lon, df.Lat, c=df[data], cmap='jet', transform=ccrs.PlateCarree(), linewidth=1, alpha=1.0, marker = '.', s=dotsize, vmin=mincolor, vmax=maxcolor, zorder=10)
    
    if code == 'AB':
        description = '    '+antB+'    $\Delta SNR$ (dB)    '+antA
        plt.figtext( 0.08, 0.985, 'A: '+AName[-1], fontsize=7)   
        plt.figtext( 0.08, 0.96, 'B: '+Bname[-1], fontsize=7)

    elif code == 'BC':
        ref = xml.getXML('lastmatchsortdescrpta')
        Bdes = xml.getXML('descriptionb')
        Cdes = xml.getXML('lastmatchsortdescrptb')
        description = 'fm Ref:'+Cdes+'  $\Delta SNR$  '+Bdes
        plt.figtext( .08, 0.989, 'B: '+Bdes, fontsize=5)   
        plt.figtext( .08, 0.964, 'C: '+Cdes, fontsize=5)
        plt.figtext( .08, 0.939, 'Ref: '+ref, fontsize=5)    
   
    plt.colorbar(pad = .02, location = 'bottom', shrink=0.4, aspect = 40, label=description)  

    if savedplots:
        fig.savefig(filename,bbox_inches="tight",pad_inches=0.2)
        print('Saved plot:',filename)
    if inlineplots:
        plt.show()
    else:
        plt.clf() 
    f3.close()
    

def strip_path(file):
    levels = file.split('/')
    return levels[-1]  

def getInteger(strng,message):
    try:
        userInt = int(strng)
        return userInt
    except ValueError:
        while True:
            try: 
                userInt = int(input(message))
                return userInt
            except:
                print('Try again')
            
def getXMLlocal():
    global fileA, fileB
    global antA, antB
    try:
        f=open("allplot.xml", "r")
        content = f.read()
        soup = bs(content, features="html.parser")
    except:
        print("No allplot.xml found")
        return    
    tag = soup.config.files.filea    
    fileA = tag['filea']   
    fileB = soup.config.files.fileb['fileb']
    antA = soup.config.files.descriptiona['descriptiona']
    antB = soup.config.files.descriptionb['descriptionb']
    
#    print(fileA, antA, fileB, antB)

def float_try(str,default):
    try:
        a = float(str)
        return a
    except ValueError:
        return default               

            
######################### Command Parser and Main Loop  ###################################

status()

print("->HELP to see command list")

while True:  #  Main -- command parser
    filetypes = (('Text files', '*.TXT'),('All files', '*.*'),)
    stuff = ''
    try:
        cmdin = input('->')
    except KeyboardInterrupt:
        break   
    cmds = cmdin.split(' ',1)
 #   print('Length of command string:',len(cmds))   
    cmd = cmds[0]
#    print("Length of cmds:",len(cmds))    
#    print("Command: ",cmd)
#    for x in cmds:
#        print(x)
    
    if len(cmd) >0 and (cmd[0] == 'Q' or cmd[0] == 'q'):
        print('good bye')
        break
    
    elif cmd == 'FA' or cmd == 'fa':
        root = tk.Tk()
        filein = fd.askopenfilename(title='Select "A" ALL.TXT file...',filetypes=filetypes,)
        root.destroy()
        if filein != '':
          fileA = filein
          xml.updateXML('filea',fileA) 
          xml.updateXML('havematches',False)
          xml.updateXML('havesorted',False)
          havematches = False
          havesorted = False
        stuff = input('A: '+split_path(fileA,"name")+' Give Description ->')
        if stuff != '':
            antA = stuff
            xml.updateXML('descriptiona',antA)                       
        print(split_path(fileA,"name"), antA) 
        spath = split_path(fileA,"path")
        print('Path: ', spath)
 #       santA = xml.getXML('savematchsortdescrpta')
  #      if antA != santA:
        havereference = False
        xml.updateXML('havereference',False)
                    
    elif cmd == 'FB' or cmd == 'fb':
        root = tk.Tk()
        filein = fd.askopenfilename(title='Select "B" ALL.TXT file...',filetypes=filetypes,)
        root.destroy()
        if filein != '':
          fileB = filein
          xml.updateXML('fileb',fileB)         
        else:
          fileB = "ALL.txt"  
          xml.updateXML('fileb',fileB) 
          
        xml.updateXML('havematches',False)
        xml.updateXML('havesorted',False)
        havematches = False
        havesorted = False
          
        stuff = input('B: '+strip_path(fileB)+' Give Description ->')
        if stuff != '':
            antB = stuff 
            xml.updateXML('descriptionb',antB)
        print(strip_path(fileB), antB) 
        
        
    elif len(cmd) >0 and (cmd[0] == 'H' or cmd[0] == 'h'):
        print('Commands:')
        print('FA  Specify the A data file')
        print('FB  specify the B data file')
        print('CF  Combine Files - Find simultaneous matches')
        print('PO  POlar plot - process signal matches')
        print('BA  specify the Bands for data plot')
        print('AR  Animate Raw ALL.TXT file(s) on map')
        print('AB  animation of SNR A - SNR B data on map')
        print('PR  Polar plot agains a Reference')
        print('MS  Map Siganls - SNR A-B')
        print('MR  Map compared to A Reference: B-A - C-A => B-C')
        print('CX  set color scale Maximum SNR A-B')
        print('CN  set color scale Minimum SNR A-B')
        print('SX  set polarplot scale Maximum')
        print('SN  set polarplot scale Minimum')
        print('ST  status')
        print('DP  set figure and animation resolution DPI')
        print('FS  set figure size, x-inches, y-inches')
        print('FT  turn on/off curve fit, F+/F-, Fourier degree, N (int)')
        print('DT  set time between frames for animation (minutes)')
        print('IT  set data collection time for frame (minutes)')
        print('LO  specify longitude, Latitude of observer')
        print('MX  set max SNR for interval to reject signals')
        print('VW  plot view: WOrld, NAmerica, AMericas, NHemisphere')
        print('PL  Plot Output: I+ inline console; S+ saved to file;  I-, S- turn off')
        print('XM  print out XML data')
        print('XD  reset allplot.xml to defaults and quit')
        print('Q   Quit')
        print('H   Help - this info...')
        
    elif cmd == 'ST' or cmd =='st':
        status()
        
    elif cmd == 'AB' or cmd == 'ab':
        mapAB()
        
    elif cmd == 'PR' or cmd == 'pr':
        ABRef()
    
    elif cmd == 'BA' or cmd == 'ba':
        if len(cmds) > 1:
            bnd = cmds[1]
            print('Only ',bnd)
        else:
            if bnd != '':
                print('Only ',bnd)
            else:
                print('All bands')
            stuff = input('Band to plot ->')
            bnd = stuff
            if stuff != '':  
                print('Only ',bnd,' band.')   
            else:
                bnd = 'All bands'
                print(bnd)
        xml.updateXML("band",bnd)
        havemeans = False
               
    elif cmd == 'CX' or cmd == 'cx':
         if len(cmds) > 1:
             maxcolor = float_try(cmds[1],10.0)
         else:
             stuff = input('Max SNR A-B for color map ->')                
             if stuff == '':
                 maxcolor = maxsnrab
                 if maxcolor == 0:
                     print('PO the data to determine autoscale!')
             else:
                 maxcolor = float_try(stuff,10.0)
         xml.updateXML("clrscalemx",maxcolor)
         print('Color range from ',mincolor,' to ',maxcolor)
     
    elif cmd == 'CN' or cmd == 'cn':
         if len(cmds) > 1:
             mincolor =float_try(cmds[1],-10.0)
         else:
             stuff = input('Max SNR A-B for color map ->')                
             if stuff == '':
                 mincolor = minsnrab
                 if mincolor == 0:
                     print('PO the data to determine autoscale!')
             else:
                 mincolor = float_try(stuff,-10.0)
         xml.updateXML("clrscalemn",mincolor)
         print('Color range from ',mincolor,' to ',maxcolor)       
        
    elif cmd == 'SX' or cmd == 'sx':
        if len(cmds) > 1:
            maxscale = float_try(cmds[1],15.0)
        else:
            stuff = input('Max SNR A-B scale for polar plot ->')                
            if stuff == '':
                autoscale=True
            else:
                maxscale = float_try(stuff,15.0)
                autoscale = False
            
        xml.updateXML("scalemx",maxscale)
        xml.updateXML("pautoscale",autoscale)
        if autoscale:
            print('Polar plot autoscale')
        else:
            print('Polar scale:',minscale,' to ',maxscale)
    
    elif cmd == 'SN' or cmd == 'sn':
        if len(cmds) > 1:
            minscale = float_try(cmds[1],-15.0)
        else:
            stuff = input('Min SNR A-B scale for polar plot ->')                
            if stuff == '':
                autoscale=True
            else:
                minscale = float_try(stuff,-15.0)
                autoscale = False
            
        xml.updateXML("scalemn",minscale)
        xml.updateXML("pautoscale",autoscale)
        if autoscale:
            print('Polar plot autoscale')
        else:
            print('Polar scale:',minscale,' to ',maxscale)
            
    elif cmd == 'DP' or cmd == 'dp':
        if len(cmds) > 1:
            dt = int(cmds[1])
            if dt >=50 and dt <= 1000:
                dpi = dt 
                xml.updateXML("dpi",dt)
                plt.rcParams["figure.dpi"] = dpi        
        print('Resolution: ',dpi,' DPI')   
    elif cmd == 'DT' or cmd == 'dt':
        if len(cmds) > 1:
            dt = int(cmds[1])
            if dt >=1 and dt <= 60:
                delta_t = dt
        print('Time between frames: ',delta_t,' minuntes') 
                    
        
    elif cmd == 'IT' or cmd == 'it':
        if len(cmds) > 1:
            dt = int(cmds[1])
            if dt >=1 and dt <= 60:
                int_t = dt
        print('Time between frames: ',int_t,' minuntes')
        
    elif cmd == 'LO' or cmd == 'lo':
        cmds = cmdin.split(' ')
        if len(cmds) > 2:
            stuff = cmds[1]
            num = float(stuff)
            if num >= -180. and num <= 180. :
                clon = num
                xml.updateXML('longitude',stuff) 
            stuff = cmds[2]
            num = float(stuff)
            if num >= -90.0 and num <= 90.0 :
                clat = num
                xml.updateXML('latitude',stuff) 
            print('Central Lon,Lat: ',clon,',',clat)   
                            
        else:
            print(cmds)
            print('Syntax:  LO -124.3 45.2   try again')
            
    elif cmd == 'FS' or cmd == 'fs':
         cmds = cmdin.split(' ')
         if len(cmds) > 2:
             stuff = cmds[1]
             num = float(stuff)
             if num >= 1. and num <= 10. :
                 fsx = num
                 xml.updateXML('xsize',fsx) 
             stuff = cmds[2]
             num = float(stuff)
             if num >= 1.0 and num <= 10.0 :
                 fsy = num
                 xml.updateXML('ysize',fsy) 
             print('plot size: ',fsx,'by',fsy)   
                             
         else:
             print('Syntax:  FS ',fsx,' ',fsy,'  try again')
    
    elif cmd == 'FT' or cmd == 'ft':
        if len(cmds)>1:
            stuff = cmds[1]
            if stuff == 'F+' or stuff == 'f+':
                fitdata = True
                xml.updateXML('fitdata',fitdata)
            elif stuff == 'F-' or stuff == 'f-':
                fitdata = False
                xml.updateXML('fitdata',fitdata)
            else:
                nfit = getInteger(stuff,'Order->')
                xml.updateXML('fitorder',nfit)
        print('Fit turned', onf(fitdata),' order',nfit)
        model_dict = {y: fourier_series(x, f=1, n=nfit)}
                
    elif cmd == 'MX' or cmd == 'mx':
        if len(cmds) > 1:
            MaxSig = getInteger(cmds[1],'MaxSig ->')
        havesorted = False    
        print("Reject data if maximum Signal report for timestamp is > ",MaxSig)      
            
    elif cmd == 'AR' or cmd == 'ar':
        mapum()        
        
    elif cmd == 'CF' or cmd == 'cf':
        matches()
        
    elif cmd == 'PO' or cmd == 'po':
        process() 
        
    elif cmd == 'MS' or cmd == 'ms':
        mapsigs('AB')
        
    elif cmd == 'MR' or cmd == 'mr':
        mapsigs('BC')
    
    elif cmd == 'PL' or cmd == 'pl':
        if len(cmds) > 1:
            stuff = cmds[1]
        else:
            print('Inline Plots:',inlineplots,'  Saved Plots:',savedplots)
            print("Plots: I+,I- turn on/off inline cosole plots; S+,S- turn on/off saved plots to file")
        if stuff == 'I+' or stuff =='i+':
            inlineplots = True
            xml.updateXML('inlineplots',inlineplots) 
        elif stuff == 'I-' or stuff == 'i-':
            inlineplots = False
            xml.updateXML('inlineplots',inlineplots) 
        elif stuff == 'S+' or stuff == 's+':  
            savedplots = True
            xml.updateXML('savedplots',savedplots) 
        elif stuff == 'S-' or stuff == 's-':
            savedplots = False
            xml.updateXML('savedplots',savedplots) 
        else:
            print('??')
        
        
    elif cmd == 'VW' or cmd == 'vw':
        if len(cmds) > 1:
            stuff = cmds[1]
        else:
            print("Map Views: WO world, NH N.Hemisphere, NA N.America, AM N&S America")
            print("Effects:  F+ F- Map Image, S+ S- Night Shadow")
            stuff = input('->')
        if stuff == 'F+' or stuff == 'f+':
            mapimage = True
        elif stuff == 'F-' or stuff == 'f-':
            mapimage = False
        elif stuff == 'S+' or stuff == 's+':
            shadow = True
        elif stuff == 'S-' or stuff == 's-':
            shadow = False
        else:
            view = stuff
        print(setExtent(view),'Mapimage:',mapimage)
        
        
    elif cmd == 'XM' or cmd =='xm':
        xml.readXML()
        
    elif cmd == 'XD' or cmd == 'xd':
        xml.defaultXML()
        break    
    else:
        if cmd != '':
            print('No such command:',cmd)
    
    