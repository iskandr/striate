from conv_net import ConvNet

c = ConvNet(batch_size = 32, n_colors = 3, learning_rate = 0.01)

import numpy as np 
xpos = np.random.randn(64, 32, 32, 3)
xneg = 10*xpos
x = np.array([xpos, xneg])
y = np.array(([1] * 64) + ([-1]* 64))
c.fit(x, y)
