# Additional experiments for Section 4.1
Here is a summary of additional experiments for Section 4.1 to demonstrate the performance of DCDP under low SNR (signal-to-noise ratio).

### Data generation

Consider the same setting of Section 4.1, that is we generate  i.i.d. univaraite Gaussian random variables $\\{y\_\{i\}\\}\_\{i\in[n]\}\subset  \mathbb{R}$  with $y\_\{i\} = \mu\_\{i\}^\{\*\} + \epsilon\_\{i\}$ and $\sigma\_\{\epsilon\}   =1$. We set $n=4\Delta$ where $\Delta \in \\{500,5000\\}$. The three population change points of $\\{ \mu\_\{i\}^\{\*\} \\}\_{i \in [n]}$ are set to be $\mu\_\{\eta\_\{0\}\}^\{\*\} = 0$, $\mu\_\{\eta\_\{1\}\}^\{\*\} = \delta$, $\mu\_\{\eta\_\{2\}\}^\{\*\} = 0$, $\mu\_\{\eta\_\{3\}\}^\{\*\} = \delta$, where $\eta\_\{k\} = k\Delta + \delta\_\{k\}$ with $\delta\_\{k\}\sim Unif[-\frac\{3\}\{10\}\Delta,\frac\{3\}\{10\}\Delta]$ for $k = 1,2,3$. 

Again the error in following figures refers to the Hausdorff distance $H(\{\widehat{\eta}\_\{k\}\}\_\{k\in [\widehat{K}]\},\{\eta\_\{k\}\}\_\{k\in [K]\})$ as a measurement of the difference between the estimators and the true change points.

### Results


<img src="https://github.com/MountLee/DCDP/figures/files/DCDP_Q100_error_vs_delta.png">
