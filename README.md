# Forecasting with feedforward and LSTM neural networks

## Paper & assoicated research
Read <a href="https://github.com/ShrutiAppiah/crypto-forecasting-with-neuralnetworks/blob/master/Catastrophes%20in%20volatile%20financial%20markets.pdf"> paper. </a>

## Highlights
### Noise can sometimes be good!
- Noise helps optimizers escape saddle points and local maxima/minima

### Vanishing gradients are a problem. LSTM units save the day.
- LSTM (Long Short-term Memory) neural networks are mindful of long-term dependencies. They remember things from the past just like your girlfriend does. Read more about <a href="https://en.wikipedia.org/wiki/Gradient_descent"> gradient descents. </a> 

### Adam optimizers can recalculate neuron weights based on both first and second order moments
- Adam optimizers combine Adaptive Gradient (AdaGrad) and Root Mean Square Propogation (RMS Prop) calculators. 
- In a distribution, the first-oder moment is the mean. The second-order moment is the variance.
- AdaGrad is great at handling sparse gradients. It calculates second-order moments based on multiple past gradients.
- RMSProp is based solely on first-order moments i.e means.
- Combined, the Adam Optimizer produces more sensible learning rates in each iteration.

## Overview
<div align="center">
		<img src="Slide1.jpeg" alt="poster">
		<br>
		<br>
</div>
