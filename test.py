import numpy as np
import matplotlib.pyplot as plt
import correlate_vmi_pico as cvp

timepix_file=r"J:\ctgroup\Edward\DATA\VMI\20240730\firstNanoData_06_30min.h5"
out_file=r"J:\ctgroup\Edward\DATA\VMI\20240730\firstNanoData_Vth600mV_30min_combined.h5"
pico_file=r"J:\ctgroup\Edward\DATA\VMI\20240730\firstNanoData_Vth600mV_30min.h5"

x,y,t,tot,laser_times,ion_times=cvp.load_timepix_data(timepix_file)
np_hits=cvp.find_potential_vmi_np_hits(ion_times)
pico_hits=cvp.load_pico_data(pico_file)
