# Additional experiments for Section 4.1
Here is a summary of additional experiments for Section 4.1 to demonstrate the performance of DCDP under low SNR (signal-to-noise ratio).

### Data generation

Consider the same setting of Section 4.1, that is we generate  i.i.d. univaraite Gaussian random variables $\\{y\_\{i\}\\}\_\{i\in[n]\}\subset  \mathbb{R}$  with $y\_\{i\} = \mu\_\{i\}^\{\*\} + \epsilon\_\{i\}$ and $\sigma\_\{\epsilon\}   =1$. We set $n=4\Delta$ where $\Delta \in \\{500,5000\\}$. The three population change points of $\\{ \mu\_\{i\}^\{\*\} \\}\_{i \in [n]}$ are set to be $\mu\_\{\eta\_\{0\}\}^\{\*\} = 0$, $\mu\_\{\eta\_\{1\}\}^\{\*\} = \delta$, $\mu\_\{\eta\_\{2\}\}^\{\*\} = 0$, $\mu\_\{\eta\_\{3\}\}^\{\*\} = \delta$, where $\eta\_\{k\} = k\Delta + \delta\_\{k\}$ with $\delta\_\{k\}\sim Unif[-\frac\{3\}\{10\}\Delta,\frac\{3\}\{10\}\Delta]$ for $k = 1,2,3$. 

Again the error in following figures refers to the Hausdorff distance $H(\{\widehat{\eta}\_\{k\}\}\_\{k\in [\widehat{K}]\},\{\eta\_\{k\}\}\_\{k\in [K]\})$ as a measurement of the difference between the estimators and the true change points. All results below are based on 100 trials.

### Results

We set the number of grid points $\mathcal{Q}=100$, and the result is (where the curve is the average and the shaded area corresponds to upper and lower 0.1 quantiles)

![DCDP_Q100_error_vs_delta](https://github.com/MountLee/DCDP/blob/main/figures/files/DCDP_Q100_error_vs_delta.png)

Since the data is univariate, $\delta$ is equal to $\kappa$, the jump size of signals. To see if the localization error when $\delta=0.5, \Delta=500$ is reasonable, we also check the simplest setting, where $n=1000$ and it is known that there is only one true change point at $\eta^* = 500$. In this setting, the best choice is to simply pick the extreme point of the CUSUM statistic as the estimated change point, which leads to the following result:

![single_cp](https://github.com/MountLee/DCDP/blob/main/figures/files/single-cp.png)

It can be seen that with similar SNR, the localization error of DCDP under the (much more difficult) multiple change point setting is only twice of the error of the most powerful method in the simplest case. This demonstrate that DCDP performs well under low SNR scenarios. 

Furthermore, if we set $\mathcal{Q}=n=2000$ in the setting $\Delta = 500$, then the "divide step" corresponds to the vanilla DP and "DCDP" corresponds to vanilla DP + local refinement. Theoretically, this would lead to more accurate estimates, but with a much higher computational price. The result for $\delta = 0.5, 0.75$ is:

![DP](https://github.com/MountLee/DCDP/blob/main/figures/files/DP.png)

It can be seen that the improvement on the localization error against that of $\mathcal{Q} =100$ is fairly small, while the run time is more than 200 times longer. This demonstrates that DCDP is efficient and accurate, even when SNR is low.
