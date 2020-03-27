import numpy as np 
import matplotlib.pyplot as plt  

px = np.linspace(0.01, 0.99, 1000) 
y = px * np.log2(1/px) + (1-px) * np.log2(1/(1-px)) 

plt.plot(px, y)  
plt.show() 