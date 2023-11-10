import numpy as np
import matplotlib.pyplot as plt

def get_coef(size, time):
    size_log = np.log(size)
    time_log = np.log(time)
                                             
    # plt.plot(size_log, time_log)
    print(np.polyfit(size_log, time_log, 1))

def main():
    size = np.array([512, 1024, 2048, 4096])
    time = np.array([200, 1385, 9007, 64440])

    just_opt_size = np.array([512, 1024, 2048, 4096])
    just_opt_time = np.array([23, 188, 1311, 10314])
    

main()
