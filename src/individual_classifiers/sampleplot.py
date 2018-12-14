#use these two lines FIRST if you get weird errors.
#They tell python what window server to use to display plots,
#which for some reason doesn't always default correctly on Macs.
import matplotlib as mpl
mpl.use('TkAgg')
#always use this to import for plotting.
#may have to run "pip install matplotlib" in commandline.
import matplotlib.pyplot as plt

#example arrays
X = range(0,10)
Y1 = range(1,11)
Y2 = []
for i in range(len(X)):
    Y2.append(X[i]*2)

#basicplotting example
plt.plot(X,Y1)

#plotting a dashed green line with legend label
plt.plot(X,Y2,color='g',linestyle='-', label="my line")
plt.legend()

#setting labels and legends
plt.title("my plot")
plt.xlabel("X")
plt.ylabel("Y")

#actually make a window pop up
#note that the program hangs until you close the window
plt.show()

#save the figure
plt.savefig("myplot.png")

