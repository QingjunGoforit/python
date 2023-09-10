import pyvisa as visa
import matplotlib.pyplot as plt
import datetime
import pandas as pd 
import time
import numpy as np
import nidaqmx
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import inspect
import h5py
import os 

Direc = 'C:\\Users\\labusers\\Desktop\\PsiQuantum\\Data'

### GENERIC FUNCTIONS: 

def filename(ethetaa: float, ethetas: float, dthetaa: float, dthetas: float, setting: str, code: str) -> (str):
    """
    setting: additional measurement description 
    ethetaa: encoding theta a 
    ethetas: encoding theta s 
    dthetaa: decoding theta a 
    dthetas: decoding theta s 
    code: encoding # decoding # 
    
    returns filename 
    """
    ea = str(ethetaa).replace(".", "p")
    es = str(ethetas).replace(".", "p")
    da = str(dthetaa).replace(".", "p")
    ds  = str(dthetas).replace(".", "p")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
    file = timestamp + '_ENCthetaA' + ea + 'V' + 'thetaS' + es + 'V_' + 'DECthetaA' + da + 'V' + 'thetaS' + ds + 'V_'  + setting + '_' + code +  '.csv'
    return file

def hdf5file(setting :str, pg: int) -> (str): 
    """
    setting: additional measurement description 
    pg: page number in lab notebook 
    
    returns filename
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H_%M_%S")
    file = timestamp + '_36Points_' + setting + '_pg' + str(pg) + '.hdf5'
    return file

def makePath(index: int, filelist: list) -> (str):
    path = Direc + '\\' + filelist[index][1] 
    return path

def printFilelist(): 
    files = os.listdir(Direc)
    filelist = enumerate(files, 0) 
    for index, value in filelist: 
        print("{}. {} \n".format(index, value))
    return list(enumerate(files, 0))


def iforgot(): 
    """ 
    returns what global variable corresponds to which phase shifter 
    """ 
    
    print("Its ok! The setup is kind of hairy anyways... \n") 
    print("vnew1 corresponds to theta a on the decoder (so the phase shifter for the MZI). this corresponds to 'Dev1/ao1' \n") 
    print("vnew2 corresponds to theta s on the decoder (so the phase shifter in parallel with the delay line). this corresponds to 'Dev1/ao2' \n") 
    print("vnew11 corresponds to theta a on the encoder (so the phase shifter for the MZI). this corresponds to 'Dev1/ao11' \n") 
    print("vnew12 corresponds to theta s on the encoder (so the phase shifter in parallel with the delay line). this corresponds to 'Dev1/ao12' \n") 
    
    
### SCB68A FUNCTIONS: 

def setZero(devchan: str):
    """
    devchan: 'Dev#/ao#'
    """
    with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(devchan)
            task.write([0], auto_start = True)
            
def setVoltage(volt: float, devchan: int):
    """
    volt: voltage 
    devchan: 'Dev#/ao#'
    """
    with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(devchan)
            task.write([volt], auto_start = True)

            
def setAllZero():
    setZero("Dev1/ao1")
    setZero("Dev1/ao2")
    setZero("Dev1/ao3")
    setZero("Dev1/ao4")
    setZero("Dev1/ao11")
    setZero("Dev1/ao12")
    setZero("Dev1/ao13")
    setZero("Dev1/ao14")

    
def setPSJZero():
    setZero("Dev1/ao1")
    setZero("Dev1/ao2")
    setZero("Dev1/ao3")
    setZero("Dev1/ao4")

    
def setPSIZero():
    setZero("Dev1/ao11")
    setZero("Dev1/ao12")
    setZero("Dev1/ao13")
    setZero("Dev1/ao14")


def climbVoltage(start: float, end: float , step: float , hold: int, devchan: str):
    """
    start: starting voltage (remember to divide if going through gain stage) 
    end: ending voltage, inclusive! should be voltage you want to end at 
    step: step size voltage 
    hold: hold time 
    devchan: "Dev#/ao#" 
    """
    voltage = [i for i in np.arange(start, end + step, step)]
    for i in range(len(voltage)): 
        with nidaqmx.Task() as task:
            task.ao_channels.add_ao_voltage_chan(devchan)
            task.write([voltage[i]], auto_start=True)
            time.sleep(hold)


def initializePSCOAXJ():
    #### START AND END IS INCLUSIVE, YOU'RE FINE DON'T WORRY. We checked this. 
    climbVoltage(0, 40/30, 5/30, 2, "Dev1/ao1")
    climbVoltage(0, 40/30, 5/30, 2, "Dev1/ao2")
    climbVoltage(0, 40/30, 5/30, 2, "Dev1/ao3")
    climbVoltage(0, 40/30, 5/30, 2, "Dev1/ao4")
    time.sleep(5) 
    climbVoltage(40/30, 15/30, -5/30, 2, "Dev1/ao1")
    climbVoltage(40/30, 15/30, -5/30, 2, "Dev1/ao2")
    climbVoltage(40/30, 15/30, -5/30, 2, "Dev1/ao3")
    climbVoltage(40/30, 15/30, -5/30, 2, "Dev1/ao4")
    print('Successfully Initialized (Poled) PS COAX J') 
    
    
def initializePSCOAXI():
    #### START AND END IS INCLUSIVE, YOU'RE FINE DON'T WORRY. We checked this. 
    climbVoltage(0, 40/30, 5/30, 2, "Dev1/ao11")
    climbVoltage(0, 40/30, 5/30, 2, "Dev1/ao12")
    climbVoltage(0, 40/30, 5/30, 2, "Dev1/ao13")
    climbVoltage(0, 40/30, 5/30, 2, "Dev1/ao14")
    time.sleep(5) 
    climbVoltage(40/30, 15/30, -5/30, 2, "Dev1/ao11")
    climbVoltage(40/30, 15/30, -5/30, 2, "Dev1/ao12")
    climbVoltage(40/30, 15/30, -5/30, 2, "Dev1/ao13")
    climbVoltage(40/30, 15/30, -5/30, 2, "Dev1/ao14")
    print('Successfully Initialized (Poled) PS COAX I') 
    
 

    
### DATA VISUALIZATION AND ANALYSIS 

def plotMatrix(a: list, start: int, stop: int, yhalf: float, f: dict, sigfigs: int): 
    """
    a: filelist 
    start: starting file index
    stop: stopping file index, inclusive! should be file you want to end at 
    yhalf: max ylim in mV for plotting
    f: dictionary of fidelities 
    sigfigs: number of sig figs
    """
    fig, axs = plt.subplots(6, 6, figsize = (20, 15))
    for i in range(start, stop + 1): 
        path = makePath(i, a)
        df = pd.read_csv(path, header = 0, sep = ',')
        e, d = path.split("_")[-1].split(".")[0].split("E")[-1].split("D")
        
        axs[int(e)-1][int(d)-1].plot(df['Time'], df['Top']*1000, color = 'tab:green', label = 'F = ' + str(np.round(f['E' + e + 'D' + d], sigfigs)))
        axs[int(e)-1][int(d)-1].set_ylim(bottom = -yhalf, top = yhalf)
        axs[int(e)-1][int(d)-1].legend(loc='upper left')
        
        axstwin = axs[int(e)-1][int(d)-1].twinx()
        
        axstwin.plot(df['Time'], df['Bottom']*1000, color = 'tab:blue', label = 'F = ' + str(np.round(f['E' + e + 'D' + d], sigfigs)))
        axstwin.set_ylim(bottom = -0.5, top = 2*yhalf)
    
    for ax in axs.flat:
        ax.set(xlabel = 'Time (ns)', ylabel = 'Voltage (mV)')
    
    plt.tight_layout()

    
def analyzeData(a: list, start: int, stop: int, h: int) -> (dict, dict): 
    """
    a: filelist
    start: starting file index
    stop: stopping file index, inclusive! should be file you want to end at
    h: half of the total number of indices to take for summing 
    
    returns: dictionary of peaks, dictionary of sums 
    """
    #250 points total is 195 ps 
    #500 points total is 390 ps 
    
    peaks = {}
    sums = {}
    for i in range(start, stop+1): 
        path = makePath(i, a)
        df = pd.read_csv(path, header = 0, sep = ',')
        condition = path.split("_")[-1].split(".")[0]



        if df['Bottom'].idxmax() not in range(int(len(df['Time'])/2 - 20), int(len(df['Time'])/2 + 20)): 
            print(condition + ': Bottom too small, take corresponding index') 
            sumbottom = np.sum(df['Bottom'][(df['Top'].idxmax() - h):(df['Top'].idxmax() + h)])
            pkbottom = df['Bottom'][df['Top'].idxmax()]
            sums[condition] = {'Top': np.sum(df['Top'][(df['Top'].idxmax() - h):(df['Top'].idxmax() + h)]), 'Bottom': sumbottom}
            peaks[condition] = {'Top': df['Top'].max(), 'Bottom': pkbottom}

        if df['Top'].idxmax() not in range(int(len(df['Time'])/2 - 20), int(len(df['Time'])/2 + 20)):                 
            print(condition + ': Top too small, take corresponding index') 
            sumtop = np.sum(df['Top'][(df['Bottom'].idxmax() - h):(df['Bottom'].idxmax() + h)])
            pktop = df['Top'][df['Bottom'].idxmax()]
            sums[condition] = {'Top': sumtop, 'Bottom': np.sum(df['Bottom'][(df['Bottom'].idxmax() - h):(df['Bottom'].idxmax() + h)])}
            peaks[condition] = {'Top': pktop, 'Bottom': df['Bottom'].max()}


        else: 
            peaks[condition] = {'Top': df['Top'].max(), 'Bottom': df['Bottom'].max()}
            sums[condition] = {'Top': np.sum(df['Top'][(df['Top'].idxmax() - h):(df['Top'].idxmax() + h)]), 'Bottom': np.sum(df['Bottom'][(df['Bottom'].idxmax() - h):(df['Bottom'].idxmax() + h)])}


    return peaks, sums 

def fidelity(idealrho, fishroe): 
    """
    idealrho: ideal density matrix
    fishroe: experimental density matrix
    
    return: fidelity 
    """
    return np.trace(np.matmul(idealrho, fishroe)) 

def calculateFidelities(a: list, start: int, stop: int, sums: dict, flag: str) -> (dict):
    """
    a: filelist 
    start: starting file index
    stop: stopping file index, inclusive! should be file you want to end at
    sums: analyzed data, dicitionary of sums 
    flag: either 'global' or 'local' -- 'global', you divide by the largest amount of total power 
    
    return: dictionary of fidelities 
    """
    fdict = {}
    totalsum = {}
    
    if flag == 'global': 
        for i in range(start, stop+1): 
            path = makePath(i, a)
            df = pd.read_csv(path, header = 0, sep = ',')
            code = path.split("_")[-1].split(".")[0]

            top = np.abs(sums[code]['Top'])
            bottom = np.abs(sums[code]['Bottom'])

            totalsum[code] = top + bottom 
            maxvalue = max(totalsum.values())

        for i in range(start, stop+1): 
            path = makePath(i, a)
            df = pd.read_csv(path, header = 0, sep = ',')
            code = path.split("_")[-1].split(".")[0]

            top = np.abs(sums[code]['Top'])
            bottom = np.abs(sums[code]['Bottom'])

            ### THE CASE WHERE WE EXPECT 50:50 
            if code in ['E1D3', 'E1D4', 'E1D5', 'E1D6', 'E2D3', 'E2D4', 'E2D5', 'E2D6', 'E3D1', 'E3D2', 'E3D4', 'E3D6', 'E4D1', 'E4D2', 'E4D3', 'E4D5', 'E5D1', 'E5D2', 'E5D4', 'E5D6', 'E6D1', 'E6D2', 'E6D3', 'E6D5']: 
                idealrho = np.matrix([[0.5, 0.5], [0.5, 0.5]])
                #fishroe = np.matrix([[top/(maxvalue), np.sqrt(top*bottom)/(maxvalue)], [np.sqrt(top*bottom)/(maxvalue), bottom/(maxvalue)]])
                fdict[code] = min(top/(maxvalue), bottom/(maxvalue))/max(top/(maxvalue), bottom/(maxvalue)) #fidelity(idealrho, fishroe)

            ### THE CASE WHERE WE EXPECT 100:0 
            if code in ['E1D2', 'E2D1', 'E3D3', 'E4D4', 'E5D5', 'E6D6']: 
                idealrho = np.matrix([[1, 0],[0, 0]])
                fishroe = np.matrix([[top/(maxvalue), 0 ], [0,  bottom/(maxvalue)]])
                fdict[code] = fidelity(idealrho, fishroe) 

            ### THE CASE WHERE WE EXPECT 0:100 
            if code in ['E1D1', 'E2D2', 'E3D5', 'E4D6', 'E5D3', 'E6D4']: 
                idealrho =  np.matrix([[0, 0],[0, 1]])
                fishroe = np.matrix([[top/(maxvalue), 0 ], [0,  bottom/(maxvalue)]])
                fdict[code] = fidelity(idealrho, fishroe)
                
    elif flag == 'local': 
        
        for i in range(start, stop+1): 
            path = makePath(i, a)
            df = pd.read_csv(path, header = 0, sep = ',')
            code = path.split("_")[-1].split(".")[0]

            top = np.abs(sums[code]['Top'])
            bottom = np.abs(sums[code]['Bottom'])

            ### THE CASE WHERE WE EXPECT 50:50 
            if code in ['E1D3', 'E1D4', 'E1D5', 'E1D6', 'E2D3', 'E2D4', 'E2D5', 'E2D6', 'E3D1', 'E3D2', 'E3D4', 'E3D6', 'E4D1', 'E4D2', 'E4D3', 'E4D5', 'E5D1', 'E5D2', 'E5D4', 'E5D6', 'E6D1', 'E6D2', 'E6D3', 'E6D5']: 
                idealrho = np.matrix([[0.5, 0.5], [0.5, 0.5]])
                #fishroe = np.matrix([[top/(top + bottom), np.sqrt(top*bottom)/(top + bottom)], [np.sqrt(top*bottom)/(top + bottom), bottom/(top + bottom)]])
                fdict[code] = min(top/(top + bottom), bottom/(top + bottom))/max(top/(top + bottom), bottom/(top + bottom)) #fidelity(idealrho, fishroe)

            ### THE CASE WHERE WE EXPECT 100:0 
            if code in ['E1D2', 'E2D1', 'E3D3', 'E4D4', 'E5D5', 'E6D6']: 
                idealrho = np.matrix([[1, 0],[0, 0]])
                fishroe = np.matrix([[top/(top + bottom), 0 ], [0,  bottom/(top + bottom)]])
                fdict[code] = fidelity(idealrho, fishroe) 

            ### THE CASE WHERE WE EXPECT 0:100 
            if code in ['E1D1', 'E2D2', 'E3D5', 'E4D6', 'E5D3', 'E6D4']: 
                idealrho =  np.matrix([[0, 0],[0, 1]])
                fishroe = np.matrix([[top/(top + bottom), 0 ], [0,  bottom/(top + bottom)]])
                fdict[code] = fidelity(idealrho, fishroe)
        
    return fdict

def pretune(phaseshifter: str, holdsettings: dict, vrange, npts: int, scope): 
    """
    phaseshifter: which phaseshifter do you want to tune? (options are dthetaa, dthetas, ethetaa, ethetas) 
    holdsettings: what are the biases you want for the other phase shifters? 
    vrange: voltage range (start, inclusive stop) 
    npts: number of voltage steps 
    
    returns voltages (from vrange input), times, average voltage trace of top arm @ each voltage setting, average voltage trace of bottom arm @ each voltage setting 
    
    """
    if phaseshifter == 'dthetaa': 
        vnew2 = holdsettings['dthetas'] 
        setVoltage(vnew2, 'Dev1/ao2') 
        vnew11 = holdsettings['ethetaa']
        setVoltage(vnew11, 'Dev1/ao11') 
        vnew12 = holdsettings['ethetas']
        setVoltage(vnew12, 'Dev1/ao12') 
        devchan = 'Dev1/ao1'
        
    if phaseshifter == 'dthetas':
        vnew1 = holdsettings['dthetaa']
        setVoltage(vnew1, 'Dev1/ao1') 
        vnew11 = holdsettings['ethetaa']
        setVoltage(vnew11, 'Dev1/ao11') 
        vnew12 = holdsettings['ethetas']
        setVoltage(vnew12, 'Dev1/ao12')
        devchan = 'Dev1/ao2'
        
    if phaseshifter == 'ethetaa': 
        vnew1 = holdsettings['dthetaa']
        setVoltage(vnew1, 'Dev1/ao1')
        vnew2 = holdsettings['dthetas']
        setVoltage(vnew2, 'Dev1/ao2')
        vnew12 = holdsettings['ethetas']
        setVoltage(vnew12, 'Dev1/ao12')
        devchan = 'Dev1/ao11'
        
    if phaseshifter == 'ethetas': 
        vnew1 = holdsettings['dthetaa']
        setVoltage(vnew1, 'Dev1/ao1')
        vnew2 = holdsettings['dthetas']
        setVoltage(vnew2, 'Dev1/ao2')
        vnew11 = holdsettings['ethetaa'] 
        setVoltage(vnew11, 'Dev1/ao11')
        devchan = 'Dev1/ao12'
    
    scope.run()
    voltages = np.linspace(vrange[0], vrange[1], npts)
    
    times, dataTop, dataBottom, avgTop, avgBottom = scope.grab()
    
    avgTopMatrix = np.empty((0, len(avgTop)))
    avgBottomMatrix = np.empty((0, len(avgBottom)))
    
    for i in voltages: 
        setVoltage(i, devchan)
        times, dataTop, dataBottom, avgTop, avgBottom = scope.grab()
        avgTopMatrix = np.vstack((avgTopMatrix, avgTop))
        avgBottomMatrix = np.vstack((avgBottomMatrix, avgBottom))
    
    scope.run()
    
    return voltages, times, avgTopMatrix, avgBottomMatrix 
   


### OSCILLOSCOPE FUNCTIONS: 

class oscope: 
    def __init__(self, scope, tstep, vstepT, vstepB, pts, avg, N, flag): 
        self.scope = scope
        self.tstep = tstep
        self.vstepT = vstepT
        self.vstepB = vstepB 
        self.pts = pts 
        self.avg = avg 
        self.N = N 
        self.flag = flag 
    
    def settings(self):
        print('Oscope settings are at a time step of : ' + str(self.tstep) + ' s \n') 
        print('Channel 2 Vertical voltage step of : ' + str(self.vstepT) + ' V \n') 
        print('Channel 3 Vertical voltage step of : ' + str(self.vstepT) + ' V \n') 
        print('Total points (length of trace) : ' + str(self.pts) + '\n') 
        print('Number of acquisitons to take : ' + str(self.N) + ' + 1 \n')
        print('Flag for Plotting : ' + self.flag + '\n') 
        
    def run(self): 
        self.scope.write(':run')

    def grab(self): 
        dataTop = np.array([])
        dataBottom = np.array([])
        self.scope.write(':run')
        self.scope.write('*cls')
        ch_top = 2
        ch_bot = 3
        tRange = 10*self.tstep
        vRangeT = 8*self.vstepT
        vRangeB = 8*self.vstepB
        self.scope.query('*opc?') 
        self.scope.write('acquire:average on')
        self.scope.write(f'channel{ch_top}:range {vRangeT}')
        self.scope.write(f'channel{ch_bot}:range {vRangeB}')
        self.scope.write(f'timebase:range {tRange}')

        self.scope.write('acquire:mode rtime')
        self.scope.write(f'waveform:source channel{ch_top}')
        self.scope.write('waveform:format ascii')
        self.scope.write(f'acquire:average:count {self.avg}')
        self.scope.write('acquire:average on')
        self.scope.write(':run')

        while (int(self.scope.query(':wav:count?')) < self.avg):
            time.sleep(1)

        self.scope.query('PDER?')

        ### RUN/STOP And Grab Your Data for Top & Plot

        int(self.scope.query(':wav:count?'))
        self.scope.write(f'acquire:points {self.pts}')
        self.scope.write('digitize') 

        dataTop = np.append(dataTop, self.scope.query_ascii_values('waveform:data?', container=np.array)[0:-1])
        if self.flag == 'show': 
            plt.figure()
            plt.plot(dataTop)
            plt.title('Top Trace')


        ### While the Trace is Run/Stopped, Grab Your Data for Bottom & Plot

        self.scope.write(f'waveform:source channel{ch_bot}')
        self.scope.write('waveform:format ascii')

        while (int(self.scope.query(':wav:count?')) < self.avg):
            time.sleep(1)

        self.scope.query('PDER?')

        int(self.scope.query(':wav:count?'))
        self.scope.write(f'acquire:points {self.pts}')

        dataBottom = np.append(dataBottom, self.scope.query_ascii_values('waveform:data?', container=np.array)[0:-1])
        if self.flag == 'show': 
            plt.figure()
            plt.plot(dataBottom)
            plt.title('Bottom Trace')

        ### OK, Now that we have initial grabs -- let us get the Trace N TIMES 

        for i in range(0, self.N): 
            self.scope.write(f'waveform:source channel{ch_top}')
            self.scope.write('waveform:format ascii')
            self.scope.write(':run') 
            self.scope.write(f'acquire:points {self.pts}') 
            self.scope.write('digitize') 
            dataTop = np.vstack((dataTop, self.scope.query_ascii_values('waveform:data?', container=np.array)[0:-1]))
            self.scope.write(f'waveform:source channel{ch_bot}')
            self.scope.write('waveform:format ascii')
            self.scope.write(f'acquire:points {self.pts}')
            dataBottom = np.vstack((dataBottom, self.scope.query_ascii_values('waveform:data?', container=np.array)[0:-1]))


        ### Average Out Our N+1 TRACES 

        avgTop = np.sum(dataTop, axis = 0)/np.shape(dataTop)[0]
        avgBottom =  np.sum(dataBottom, axis = 0)/np.shape(dataBottom)[0]

        xIncrement = float(self.scope.query('waveform:xincrement?'))
        xOrigin = float(self.scope.query('waveform:xorigin?'))
        yIncrement = float(self.scope.query('waveform:yincrement?'))
        yOrigin = float(self.scope.query('waveform:yorigin?'))
        length = len(dataBottom)

        times = np.array([])
        t = np.arange(0, len(dataTop[1]))
        times = np.append(times, (t*xIncrement + xOrigin)*1e9)

        for i in range(np.shape(dataTop)[0]): 
            dataTop[i] = dataTop[i] - 0.5*(np.mean(dataTop[i][0:900]) + np.mean(dataTop[i][-900:-1]))

        for i in range(np.shape(dataBottom)[0]): 
            dataBottom[i] = dataBottom[i] - 0.5*(np.mean(dataBottom[i][0:900]) + np.mean(dataBottom[i][-900:-1]))

        avgTop = avgTop - 0.5*(np.mean(avgTop[0:900]) + np.mean(avgTop[-900:-1]))
        avgBottom = avgBottom - 0.5*(np.mean(avgBottom[0:900]) + np.mean(avgBottom[-900:-1]))
        if self.flag == 'show':
            plt.figure()
            plt.plot(times, avgTop)
            plt.title('Average Top Trace')

            plt.figure()
            plt.plot(times, avgBottom)
            plt.title('Average Bottom Trace')
            
        self.scope.write(':run')

        return times, dataTop, dataBottom, avgTop, avgBottom
        
    
    