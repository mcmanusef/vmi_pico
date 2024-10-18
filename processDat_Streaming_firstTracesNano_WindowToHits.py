# File: process_experimental_data.py

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Function Definitions
def adc2mV(intADC, range_idx, maxADC):
    channel_input_ranges = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]
    vRange = channel_input_ranges[range_idx]
    mV = (intADC.astype(float) * vRange) / maxADC
    return mV

def get_sampling_interval(res, timebase):
    if res == 8:
        if -1 < timebase < 3:
            return (2 ** timebase) / 1e9
        elif timebase > 2:
            return (timebase - 2) / 125e6
    elif res == 12:
        if 0 < timebase < 3:
            return (2 ** (timebase - 1)) / 500e6
        elif timebase > 2:
            return (timebase - 3) / 62.5e6
    elif res in [14, 15]:
        if timebase == 3:
            return 1 / 125e6
        elif timebase > 3:
            return (timebase - 2) / 125e6
    elif res == 16:
        if timebase == 4:
            return 1 / 62.5e6
        elif timebase > 4:
            return (timebase - 3) / 62.5e6
    return -1

def get_bit_depth(resolution):
    bit_depth_lst = [8, 12, 14, 15, 16, 10]
    return bit_depth_lst[resolution]

def get_last_dataset_number(file):
    with h5py.File(file, 'r') as f:
        max_dataset = 0
        for name in f.keys():
            if name.startswith('CH_A_'):

                num = int(name.split('_')[2])
                max_dataset = max(max_dataset, num)
    return max_dataset


# Params
file = r"J:\ctgroup\Edward\DATA\VMI\20240730\firstNanoData_Vth800mV_real.h5"
Vth = -0.3
saBeyondThresh = 200
saLookForward = 200
bgThresh = 0.1
saAboveBg = 15
pltTraces = True

# Grab Sampling Properties
with h5py.File(file, 'r') as f:
    SI = f['/SI'][()]
    DSI = f['/DSI'][()]
    res = get_bit_depth(f['/Resolution'][()])
    vriA = f['/VoltRngIdx_CHA'][()]
    vriB = f['/VoltRngIdx_CHB'][()]
    maxADC = f['/Max_ADC_Val'][()]
    preTrigSamp = f['/PreTrigSamp'][()]
    postTrigSamp = f['/PostTrigSamp'][()]
    numSamp = int(preTrigSamp + postTrigSamp)

# Process Data
false_positive = 0
nano_hit = 0

for i in range(1, get_last_dataset_number(file) + 1):
    trace_str = f'{i:010d}'
    pathA = f'/CH_A_{trace_str}'
    pathB = f'/CH_B_{trace_str}'
    path_timestamp = f'/DatTimestamp_{trace_str}'
    path_epochSA = f'/EpochSA_{trace_str}'

    with h5py.File(file, 'r') as f:
        Va = adc2mV(f[pathA][()], vriA, maxADC) / 1000
        Vb = adc2mV(f[pathB][()], vriB, maxADC) / 1000
        trace_timestamp = f[path_timestamp][()]
        epoch_sa_before_trace = f[path_epochSA][()]
        t = np.arange(len(Va)) * DSI

    idxs = np.where(Va < Vth)[0]
    last_idx = 0
    for idx in idxs:
        if idx > last_idx + saLookForward:
            if idx + saLookForward < len(Va):
                VaTest = Va[idx + 2: idx + saLookForward]
                VaTest = VaTest[(VaTest > bgThresh) | (VaTest < -bgThresh)]
                if len(VaTest) > saAboveBg:
                    nano_hit += 1

                    # DATA FOR EACH iToF hit + trace time_stamp from above
                    tToF = (t[idx - saBeyondThresh//2:idx + saBeyondThresh] - t[idx]) * 1E6
                    VaToF = Va[idx - saBeyondThresh//2:idx + saBeyondThresh]
                    VbToF = Vb[idx - saBeyondThresh//2:idx + saBeyondThresh]
                    epoch_sa_idx_of_vth = epoch_sa_before_trace + idx

                    if pltTraces:
                        plt.subplot(2, 1, 1)
                        plt.title(f'CHA Dataset {i} -- Nano Hit {nano_hit} -- Vth idx {idx}')
                        plt.plot(tToF, VaToF)
                        plt.ylabel("iToF Voltage (V)")
                        plt.xlabel("Time (us)")
                        plt.grid(True)

                        plt.subplot(2, 1, 2)
                        plt.title(f'CHB Dataset {i} -- Nano Hit {nano_hit} -- Vth idx {idx}')
                        plt.plot(tToF, VbToF)
                        plt.grid(True)
                        plt.xlabel("Time (us)")
                        plt.ylabel("Trigger Voltage (V)")

                        plt.show()
                else:
                    false_positive += 1
                last_idx = idx
            else:
                last_idx = idx
#%%
