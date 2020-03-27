import numpy as np  
import matplotlib.pyplot as plt  
  
plt.figure(1)  
ax = plt.subplot(111)  
x = np.linspace(0, 1, 200) 

y = (1 - x ) / x   
ax.plot(x, y)  

#y2 = np.log((1 - x ) / x )  # 不能超过0.5
#ax.plot(x, y2)  
plt.show()  