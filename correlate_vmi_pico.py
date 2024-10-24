import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import h5py
# matplotlib.use('Qt5Agg')
#set colormap
# custom_jet = plt.cm.colors.ListedColormap([[1, 1, 1, 1]] + plt.cm.jet(np.linspace(0, 1, 255)).tolist())

plt.rcParams['image.cmap'] = 'jet'

def fix_time(times,threshold=1e9):
    jump_size=-np.diff(times)
    return times+np.cumsum(np.concatenate(([0],np.where(jump_size>threshold,jump_size,0))))
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


def load_timepix_data(file: str) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    with h5py.File(file, 'r') as f:
        data={}
        for k in f.keys():
            data[k] = f[k][()]

    tdc_time=data['tdc_time']
    tdc_type=data['tdc_type']
    x,y,t,tot=data['x'],data['y'],data['toa'],data['tot']

    tdc1_start_idx= np.argwhere(tdc_type==1).flatten()
    tdc1_end_idx=np.argwhere(tdc_type==2).flatten()

    tdc1_time=tdc_time[tdc1_start_idx]
    tdc1_length=tdc_time[tdc1_end_idx]-tdc_time[tdc1_start_idx]

    print(f"{max(t)=}")

    # t=t*25/16
    t=np.unwrap(t,period=25*2**30)
    t=np.asarray(t)

    laser_times=np.unwrap(tdc1_time[tdc1_length>500],period=25*2**30)
    ion_times=np.unwrap(tdc1_time[tdc1_length<500],period=25*2**30)

    t=fix_time(t)
    laser_times=fix_time(laser_times)
    ion_times=fix_time(ion_times)

    return x,y,t,tot,laser_times,ion_times


def generate_timepix_trace(laser_times: np.ndarray, ion_times:np.ndarray, time_base:np.ndarray) -> np.ndarray:
    ion_mask=np.argwhere(np.logical_and(ion_times>time_base[0],ion_times<time_base[-1])).flatten()
    laser_mask=np.argwhere(np.logical_and(laser_times>time_base[0],laser_times<time_base[-1])).flatten()
    ion_dur=200
    laser_dur=1000
    trace=np.zeros(len(time_base))
    for ion_time in ion_times[ion_mask]:
        ion_idx=np.argwhere(np.logical_and(time_base>ion_time,time_base<ion_time+ion_dur)).flatten()
        trace[ion_idx]=1
    for laser_time in laser_times[laser_mask]:
        laser_idx=np.argwhere(np.logical_and(time_base>laser_time,time_base<laser_time+laser_dur)).flatten()
        trace[laser_idx]=1
    return trace


def find_potential_vmi_np_hits(ion_times, time_diff=5000) -> list[float]:
    candidates=ion_times[np.argwhere(np.diff(ion_times)<time_diff)].flatten()
    current_candidate=0
    np_hits=[]
    hit_count=1
    hit_counts=[]
    for candidate in candidates:
        if candidate-current_candidate<time_diff*hit_count:
            hit_count+=1
        else:
            if hit_count>1:
                np_hits.append(current_candidate)
                hit_counts.append(hit_count)
                hit_count=1
            current_candidate=candidate
    return np_hits


def load_pico_data(file: str,
                Vth = -0.3, saBeyondThresh = 200, saLookForward = 200, bgThresh = 0.1, saAboveBg = 15,
                    pltTraces = False) -> list[tuple[np.ndarray,np.ndarray,np.ndarray,float]]:
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
        initRelSec= f['/InitRelSec'][()]

    # Process Data
    false_positive = 0
    nano_hit = 0
    out=[]
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
                        out.append((tToF*1000,VaToF,VbToF,(trace_timestamp-initRelSec)*1e9))

                    else:
                        false_positive += 1
                    last_idx = idx
                else:
                    last_idx = idx
    return out



def process_pico_trace(trace: np.ndarray) -> np.ndarray:
    return np.where(np.logical_or(trace>0.5,trace<-0.2), 1, 0)


def find_pico_time(pico_trace: np.ndarray, time_base: np.ndarray,
                   laser_times: np.ndarray, ion_times: np.ndarray, tpx_np_hits: list[float],
                   time_range: tuple[float,float] = (-np.inf, np.inf),
                   plotting=False, yield_fit_quality=False) -> tuple:

    max_correlation=[0]
    max_corr_index=0
    fit_quality=0
    if plotting:
        fig,ax=plt.subplots(2,1)
    for idx,np_hit in enumerate(tpx_np_hits):
        if not time_range[0]<np_hit<time_range[1]:
            continue
        tpx_trace=generate_timepix_trace(laser_times,ion_times,time_base+np_hit)
        pico_trace=process_pico_trace(pico_trace)
        correlation=np.correlate(tpx_trace,pico_trace,mode='full')
        if plotting:
            ax[0].plot(correlation)
        if max(correlation)>max(max_correlation):
            max_correlation=correlation
            max_corr_index=idx
            fit_quality=max(correlation)/max(np.correlate(tpx_trace,tpx_trace,mode='full'))



    course_time=tpx_np_hits[max_corr_index]
    t_base=time_base-time_base[0]
    t_corr=np.linspace(-max(t_base),max(t_base),len(max_correlation))
    fine_t=t_corr[np.argmax(max_correlation).flatten()[0]]
    if plotting:
        ax[1].plot(t_base,pico_trace)
        ax[1].plot(t_base-fine_t,-generate_timepix_trace(laser_times,ion_times,time_base+course_time))
        plt.show()

    if yield_fit_quality:
        return max_corr_index, fine_t+course_time, fit_quality
    return max_corr_index, fine_t+course_time#+time_base[0]

def find_electron_hit(x:np.ndarray, y:np.ndarray, t:np.ndarray, tot:np.ndarray, laser_times:np.ndarray, ion_times:np.ndarray, pico_time:float,
                      plotting=True, width=50000) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    x_hit=x[np.argwhere(np.logical_and(t > pico_time - width, t < pico_time + width)).flatten()]
    y_hit=y[np.argwhere(np.logical_and(t > pico_time - width, t < pico_time + width)).flatten()]
    t_hit=t[np.argwhere(np.logical_and(t > pico_time - width, t < pico_time + width)).flatten()]
    tot_hit=tot[np.argwhere(np.logical_and(t > pico_time - width, t < pico_time + width)).flatten()]
    laser_hit=laser_times[np.argwhere(np.logical_and(laser_times > pico_time - width, laser_times < pico_time + width)).flatten()]
    ion_hit=ion_times[np.argwhere(np.logical_and(ion_times > pico_time - width, ion_times < pico_time + width)).flatten()]
    if plotting:
        fig=plt.figure(figsize=(10,10))
        t_hit_rel=t_hit-laser_hit[np.argwhere(laser_hit<pico_time)[-1].flatten()[0]]
        # ax[0,0].hist2d(x_hit,y_hit,bins=128,range=[[0,256],[0,256]])

        toa_hist, xe,ye=np.histogram2d(x_hit,y_hit,bins=128,weights=t_hit_rel,range=[[0,256],[0,256]])
        tot_hist,xe,ye=np.histogram2d(x_hit,y_hit,bins=128,weights=tot_hit,range=[[0 ,256],[0,256]])
        plt.sca(plt.subplot(221))
        plt.title('ToA')
        plt.imshow(toa_hist,extent=[xe[0],xe[-1],ye[0],ye[-1]],origin='lower', interpolation='nearest')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.sca(plt.subplot(222))
        plt.title('ToT')
        plt.imshow(tot_hist,extent=[xe[0],xe[-1],ye[0],ye[-1]],origin='lower', interpolation='nearest')
        plt.xlabel('x')
        plt.sca(plt.subplot(212))
        plt.title("Ion ToF Trace")
    return x_hit, y_hit, t_hit, tot_hit, laser_hit, ion_hit


def find_electron_times(t,width=40000,num=200):
    out=[]
    num_samples=1
    current_candidate=0

    t_candidates=t[np.argwhere(np.diff(t)<width).flatten()]
    for t_candidate in t_candidates:
        if t_candidate-current_candidate<width:
            if t_candidate-current_candidate<0:
                current_candidate=t_candidate
            # print(t_candidate-current_candidate)
            num_samples+=1
        else:
            if num_samples>num:
                out.append(current_candidate)
                num_samples=1
            # print(num_samples)
            current_candidate=t_candidate
    return out


def main(max_diff=1e10, out_file='test.h5', timepix_file="timepix_file.h5", pico_file="pico_file.h5"):

    x,y,t,tot,laser_times,ion_times=load_timepix_data(timepix_file)
    print("Loaded Timepix data")

    fig,ax=plt.subplots(2,2)
    ax[0,0].hist2d(x,y,bins=128)
    plt.title('x-y')
    ax[0,1].hist(t,bins=128)
    plt.title('ToA')
    ax[1,0].hist(tot,bins=128)
    plt.title('ToT')
    ax[1,1].plot(np.diff(t))
    plt.title('time diff')


    np_hits=find_potential_vmi_np_hits(ion_times)
    print(f"Found {len(np_hits)} potential VMI hits")


    pico_hits=load_pico_data(pico_file)
    print(f"Loaded {len(pico_hits)} pico hits")

    with h5py.File(out_file,'w') as f:

        for i,pico_hit in enumerate(pico_hits):
            g=f.create_group(f'pico_hit_{i}')
            # print(pico_hit)
            tToF, VaToF, VbToF, epochtime=pico_hit
            tpx_idx,pico_time=find_pico_time(VbToF,tToF,laser_times,ion_times,np_hits,
                                             time_range=(epochtime-max_diff,epochtime+max_diff), plotting=False)
            # plt.suptitle(f'Event {i}')
            print(f"Time of pico hit {i}: {pico_time/1e9:.3f} {tpx_idx=}")
            print(f"Epoch time: {epochtime/1e9:.3f}, TPX time: {np_hits[tpx_idx]/1e9:.3f}, Diff: {(epochtime-np_hits[tpx_idx])/1e9:.3f}")
            x_hit, y_hit, t_hit, tot_hit,laser_hit, ion_hit = find_electron_hit(x,y,t,tot,laser_times,ion_times,pico_time,plotting=False)
            last_laser_time=laser_hit[np.argwhere(laser_hit<pico_time)[-1].flatten()[0]]
            # plt.plot(tToF+pico_time-last_laser_time,VaToF)
            # plt.tight_layout()
            # plt.plot(tToF+pico_time-last_laser_time,VbToF)
            # plt.title(f'Pi coscope Trace')
            # plt.xlabel('Time [ns]')
            # plt.ylabel('Voltage [V]')
            # plt.grid()
            # plt.suptitle(f'Event {i}')

            g.create_dataset('x',data=x_hit)
            g.create_dataset('y',data=y_hit)
            g.create_dataset('t',data=t_hit-last_laser_time)
            g.create_dataset('tot',data=tot_hit)
            g.create_dataset('laser',data=last_laser_time)
            g.create_dataset('ion',data=ion_hit-last_laser_time)

            g.create_dataset('pico_t',data=tToF+pico_time-last_laser_time)
            g.create_dataset('pico_ion_trace',data=VaToF)
            g.create_dataset('pico_fingerprint',data=VbToF)


def calculate_mean(xs, ys, ts, size=128):
    i_vals, j_vals = np.array(xs).astype(int) // 2, np.array(ys).astype(int) // 2
    # print(f"{i_vals=}",f"{j_vals=}")
    sum_vals = np.zeros((size, size))
    count_vals = np.zeros((size, size))
    np.add.at(sum_vals, (i_vals, j_vals), ts)
    np.add.at(count_vals, (i_vals, j_vals), 1)
    return np.divide(sum_vals, count_vals, out=np.zeros_like(sum_vals), where=count_vals != 0)

def make_plot(xs,ys,ts,tots,pico_ts,pico_ion_traces):
    fig=plt.figure(figsize=(10,10))
    ax=plt.subplot(221)
    toa_data=calculate_mean(xs,ys,ts)
    plt.imshow(toa_data,extent=[0,256,0,256],origin='lower',interpolation='nearest')
    plt.title('ToA')
    plt.xlabel('x')
    plt.ylabel('y')
    ax=plt.subplot(222)
    tot_data=calculate_mean(xs,ys,tots)
    plt.imshow(tot_data,extent=[0,256,0,256],origin='lower',interpolation='nearest')
    plt.title('ToT')
    plt.xlabel('x')
    plt.subplot(212)
    min_t=min(min(pico_t) for pico_t in pico_ts)
    max_t=max(max(pico_t) for pico_t in pico_ts)
    dt=pico_ts[0][1]-pico_ts[0][0]
    full_t=np.arange(min_t,max_t,dt/100)
    pico_traces_interp=[np.interp(full_t,pico_t,pico_ion_trace) for pico_t,pico_ion_trace in zip(pico_ts,pico_ion_traces)]
    mean_pico_trace=np.mean(pico_traces_interp,axis=0)
    plt.plot(full_t,mean_pico_trace)
    plt.title('PicoScope Ion Trace')
    plt.xlabel('Time [ns]')
    plt.ylabel('Voltage [V]')
    plt.grid()

def flatten_list(lst):
    return list(itertools.chain(*lst))


if __name__ == '__main__':
    # timepix_file=r"J:\ctgroup\Edward\DATA\VMI\20240730\firstNanoData_06_30min.h5"
    # out_file=r"J:\ctgroup\Edward\DATA\VMI\20240730\firstNanoData_Vth600mV_30min_combined.h5"
    # pico_file=r"J:\ctgroup\Edward\DATA\VMI\20240730\firstNanoData_Vth600mV_30min.h5"

    timepix_file=r"/mnt/NAS/ctgroup/Edward/DATA/VMI/20240730/firstNanoData_06_30min.h5"
    pico_file=r"/mnt/NAS/ctgroup/Edward/DATA/VMI/20240730/firstNanoData_Vth600mV_30min.h5"
    out_file=r"/mnt/NAS/ctgroup/Edward/DATA/VMI/20240730/firstNanoData_Vth600mV_30min_combined.h5"
    main(timepix_file=timepix_file,pico_file=pico_file,out_file=out_file,max_diff=2e20)

    # with h5py.File(out_file,'r') as f:
    #     xs=[f[k]['x'][()] for k in f.keys()]
    #     ys=[f[k]['y'][()] for k in f.keys()]
    #     ts=[f[k]['t'][()] for k in f.keys()]
    #     tots=[f[k]['tot'][()] for k in f.keys()]
    #     n_elecs=[len(x) for x in xs]
    #     n_elecs=np.array(n_elecs)
    #
    #     pico_ts=[f[k]['pico_t'][()] for k in f.keys()]
    #     pico_ion_traces=[f[k]['pico_ion_trace'][()] for k in f.keys()]
    #
    # make_plot(flatten_list(xs),flatten_list(ys),flatten_list(ts),flatten_list(tots),pico_ts,pico_ion_traces)
    # plt.suptitle('Mean of all events')
    #
    # plt.figure()
    # plt.hist(n_elecs,bins=100)
    # cutoffs=[0,600,2000,10000,15000]
    # for cutoff in cutoffs:
    #     plt.axvline(cutoff,color='r')
    # plt.title('Number of electrons in each event')
    #
    # for i in range(len(cutoffs)-1):
    #     mask=np.logical_and(n_elecs>cutoffs[i],n_elecs<cutoffs[i+1])
    #     make_plot(flatten_list([x for x, m in zip(xs,mask) if m]),flatten_list([y for y, m in zip(ys,mask) if m]),
    #               flatten_list([t for t, m in zip(ts,mask) if m]),flatten_list([tot for tot, m in zip(tots,mask) if m]),
    #               [pico_t for pico_t, m in zip(pico_ts,mask) if m],[pico_ion_trace for pico_ion_trace, m in zip(pico_ion_traces,mask) if m])
    #     plt.suptitle(f'Mean of events with {cutoffs[i]} < n_electrons < {cutoffs[i+1]}')


#%%

#%%
