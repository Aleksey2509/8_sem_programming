
import matplotlib.pyplot as plt

x = [1, 4]
time = [1.047940, 0.606802]

eff = [time[0] / (time[i] * x[i]) for i in range (len(x))]

plt.plot(x, eff)
  
# naming the x axis
plt.xlabel('number of cores')
# naming the y axis
plt.ylabel('time spent')
  
# giving a title to my graph
plt.title('Efficiency (# of cores)')
  
# function to show the plot
plt.show()
