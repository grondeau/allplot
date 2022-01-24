# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 21:09:52 2021

@author: Gary

Looking for a program to generate HDSDR record Schedule
"""

#  160  80  40  30  20  17  15  12  10

#rough transitions:  Midnight -- 160 80 40 30  6:00 AM 
#                    6:00 AM  -- 160 80 40 30 20  7:00 AM
#                    7:00 AM  -- 80  40 30 20 17  8:30 AM
#                    8:30 AM --  40 30 20 17 15  10:30 AM
#                   10:30 AM --  30 20 17 15 12  11:30 AM
#                   11:30 AM --  20 17 15 12 10  3:00 PM
#                    3:00 PM ==  40 30 20 17 15  4:30 PM
#                    4:30 PM ==  80 40 30 20 17  6:00 PM
#                    6:00 PM --  160 80 40 30 20 8:00 PM
#                    8:00 PM --  160 80 40 30   Midnight

import pandas as pd
import datetime as dt

file = 'sked.csv'

header = ['SrtTime', 'StpTime','Daily','LO_f','Tune_f','Mode']

Adate = "2020-01-01"

ts = pd.DataFrame({ 
    'time':['00:00', '06:00','07:00','08:30','10:30','11:30','15:00','16:30','18:00','20:00','23:57'],
    'bands':[['160m','80m','40m'],
             ['160m','80m','40m'],
             ['160m','80m','40m'],
             ['160m','80m','40m'],
             ['160m','80m','40m'],
             ['160m','80m','40m'],
             ['160m','80m','40m'],
             ['160m','80m','40m'],
             ['160m','80m','40m'],
             ['160m','80m','40m'],
             ['160m','80m','40m']]
    })

"""


'bands':[['160m','80m'],
         ['160m','80m'],
         ['160m','80m'],
         ['160m','80m'],
         ['160m','80m'],
         ['160m','80m'],
         ['160m','80m'],
         ['160m','80m'],
         ['160m','80m'],
         ['160m','80m'],
         ['160m','80m']]
   

#  Winter selection  
    'bands':[['160m','80m','40m','30m',],
             ['160m','80m','40m','30m','20m'],
             ['80m','40m','30m','20m','17m','15m'],
             ['40m','30m','20m','17m','15m','12m'],   
             ['30m','20m','17m','15m','12m','10m'],
             ['20m','17m','15m','12m','10m'],
             ['40m','30m','20m','17m','15m'],
             ['80m','40m','30m','20m','17m'],
             ['160m','80m','40m','30m','20m'],
             ['160m','80m','40m','30m',],
             ['160m','80m','40m','30m',]]
"""    
    
                 
ts['time'] = pd.to_datetime(Adate+' '+ts['time'])

dtmin = 2 
LOoffset = 8000
mode = 'USB'

#print(ts.time[0]+dt.timedelta(minutes=dtmin))

def excel_date(date1):
    temp = dt.datetime(1900, 1, 1)    # Note, not 31st Dec but 30th!
    delta = date1 - temp
    return float(delta.days) + (float(delta.seconds) / 86400)

def freq_from_band(f):
    if f=='160m':
        v = 1840000
    elif f=='80m':
        v = 3573000
    elif f=='40m':
        v = 7074000
    elif f=='30m':
        v = 10136000
    elif f=='20m':
        v = 14074000
    elif f=='17m':
        v = 18100000
    elif f=='15m':
        v = 21074000
    elif f=='12m':
        v = 24915000  
    elif f=='10m':
        v = 28074000   
    else:
        v = 14074000
        print('bad band', f)
    return v

#  number of bands,  time/band, transition times of day.  
def SkedBuild():
   lines = []  
   
   header=['time','stop','one','LOf','F','mode','A','B','C','D','E'] 
   
   i=0
   j=0
   
   time = ts.time[0]
   lines.append((excel_date(ts.time[i]),excel_date(ts.time[i]+dt.timedelta(seconds=2)),1,
              freq_from_band(ts.bands[i][j])-LOoffset,freq_from_band(ts.bands[i][j]),
              mode, 1,0,0,0,1))  
   
   while True:  
       time = ts.time[i]
       print('time period:',time)      
       while time<ts.time[i+1]:
           time += dt.timedelta(minutes=dtmin)
           j += 1
           if j>=len(ts.bands[i]):
              j=0 
           
           lines.append((excel_date(time),excel_date(time+dt.timedelta(seconds=2)),1,
                 freq_from_band(ts.bands[i][j])-LOoffset,freq_from_band(ts.bands[i][j]),
                 mode, 1,0,0,0,1))
       
       if i<len(ts.time)-2 :
           print(i, len(ts.time)-2)
           i += 1
           
       else:    
           break
     
   sked = pd.DataFrame(lines, columns = header)
     
#   print(sked)
     
   sked.to_csv(file, header=False,index=False)   

#   sked['time'] = pd.to_datetime(sked['time'],unit='D') 
   
#   sked.to_csv('debug.csv')

######################### Command Parser and Main Loop  ###################################
while True:  #  Main -- command parser
    filetypes = (('Text files', '*.TXT'),('All files', '*.*'),)
    cmdin = input('->')
    cmds = cmdin.split(' ',1)
 #   print('Length of command string:',len(cmds))   
    cmd = cmds[0][0:2]
#    print("Length of cmds:",len(cmds))    
#    print("Command: ",cmd)
#    for x in cmds:
#        print(x)
    
    if cmd == 'Q' or cmd == 'q' or cmd == 'QU' or cmd == 'qu':
        print('I quit')
        break
    
    elif cmd == 'FI' or cmd == 'fi':
        if len(cmds)>1:
            file = cmds[1]            
        else:
            stuff = input('Output File Name ->')
            if stuff != '':
                file = stuff            
        if len(file.split(sep='.'))<2:
            file = file+'.csv'                
        print('Output to:',file)
        
       
    elif len(cmd) >0 and (cmd[0] == 'H' or cmd[0] == 'h'):
        print('Commands:')
        print('SK  Build the Sked file')
        print('SS  Show the Sked plan')
        print('FI  Filename for schedule')
        print('DT  Time per band change (minutes)')
        print('H  Help - this info...')
    
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
                bnd = 'All Bands'
                print(bnd)
      
    elif cmd == 'SS' or cmd == 'ss':    
        print(ts)
        
    elif cmd == 'SK' or cmd == 'sk':
        SkedBuild()
           
    elif cmd == 'DT' or cmd == 'dt':
        if len(cmds) > 1:
            dtmin = int(cmds[1])
        else:
            stuff = input('Time per band (minutes) ->')  
            if stuff != '':
                dtmin = int(stuff)
        print('Time per band:',dtmin,' minutes')
       
            
    else:
        if cmd != '':
            print('No such command:',cmd)