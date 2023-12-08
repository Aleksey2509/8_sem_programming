
import matplotlib.pyplot as plt

x = [1, 4]
time = [0.017940, 0.007802]

eff = [time[0] / (time[i]) for i in range (len(x))]

plt.plot(x, eff)
  
# naming the x axis
plt.xlabel('number of cores')
# naming the y axis
plt.ylabel('time spent')
  
# giving a title to my graph
plt.title('Speed up (# of cores)')
  
# function to show the plot
plt.show()
