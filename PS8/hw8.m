Xplan: a 728 × 97 matrix of real data, where each row is a spike count vector across 97
simultaneously-recorded neurons. We are providing you here with the spike counts 
which are taken in a 200 ms bin during the plan period.
There are 91 trials for each of 8 reaching angles, for a total of 728 trials. 
Trials 1 to 91 correspond to reaching angle 1, trials 92 to 182 correspond to reach angle 2, etc.
Xsim: a 8 × 2 matrix of simulated data, where each row is a data point.

1. Visualization of high-dimensional neural activity using PCA
Here, we will apply PCA to these same data (Xplan) to gain some intuition about why we were
able to classify so well.
The data points are xn ∈ RD (n = 1, ..., N),
where D = 97 is the data dimensionality and N = 728 is the number of data points.

(a) Plot the square-rooted eigenvalue spectrum. If you had to identify an
elbow in the eigenvalue spectrum, how many dominant eigenvalues would there
be? What percentage of the overall variance is captured by the top 3 principal
components? Don't use the built-in MATLAB function for this.

(b) For the purposes of visualization, we’ll consider the PC space defined
by the top M = 3 eigenvectors. Project the data into the three-dimensional PC
space. Plot the projected points in Matlab using ‘plot3’, and color each dot
appropriately according to reaching angle (there should be a total of 728 dots).

(c) Define a matrix UM ∈ RD×M containing the top three eigenvectors
(i.e., PC directions), where UM(d, m) indicates the contribution of the dth neuron
to the mth principal component. Show the values in UM by calling ‘imagesc(UM )’.
(Note: Also call ‘colorbar’ to show the scale.)
Are there are any obvious groupings among the neurons in each column of UM?



2. From PCA to PPCA and FA
We will compare PCA, PPCA, and FA on a toy problem using the data in Xsim. The
data points are xn ∈ RD (n = 1, ..., N), where D = 2 is the data dimensionality and
N = 8 is the number of data points. We will project the data into a M = 1 dimensional
space.

write a MATLAB script based on the requirements below:

Xsim: a 8 × 2 matrix of simulated data, where each row is a data point.
where D = 2 is the data dimensionality and
N = 8 is the number of data points. We will project the data into a M = 1 dimensional
space.

(a) Create one plot containing all of the following for PCA:
• Plot each data point xn as a black dot in a two-dimensional space.
• Plot the mean of the data µ as a big green point.
• Plot the PC space defined by u1 as a black line. (Hint: This line should pass
through µ.)
• Project each data point into the PC space, and plot each projected data point
xn_pc as a red dot. (Hint: The projected points should lie on the u1 line.)
• Connect each data point xn with its projection xn_pc using a red line. (Hint:
The red lines should be orthogonal to the PC space. To see this, you will
need to call ‘axis equal’ in Matlab.)

(b) Implement the EM algorithm for PPCA in Matlab, and run the
algorithm on the data in Xsim. Plot the log data likelihood (sum of log P(xn))
versus EM iteration. (Hint: The log data likelihood should increase monotonically
with EM iteration. You should run enough EM iterations to see a long, flat
plateau.)

(c) Using the parameters found in part (b), what is the PPCA covariance
(WW^T +σ^2I)? If you did part (b) correctly, the PPCA covariance should be very
similar to the sample covariance. check the covariances using the built-in.

(d) Create one plot containing all of the following for PPCA:
• Plot each data point xn as a black dot in a two-dimensional space.
• Plot the mean of the data µ as a big green point.
• The PC space found by PPCA is defined by W, which in this case is a two-dimensional vector.
Check that PC space defined by W is identical to that
found by PCA. Plot the PC space as a black line.
• Project each data point into the PC space using PPCA, and plot each projected data point xn_pc = W E[zn | xn] + µ as a red dot. (Hint: The projected
points should lie on the line defined by W.)
• Connect each data point xn with its projection xn_pc using a red line. Why are
the red lines no longer orthogonal to the PC space?

(e) Implement EM algorithm for FA. You should be able to do this with
only a small modification to your PPCA code. Run the algorithm on the data in
Xsim. Plot the log data likelihood versus EM iteration.

(f) Using the parameters found in part (e), what is the FA covariance
(WWT + Ψ)? If you did part (e) correctly, the FA covariance should be very
similar to the sample covariance. check the covariances.

(g) Create one plot containing all of the following for FA:
• Plot each data point xn as a black dot in a two-dimensional space.
• Plot the mean of the data µ as a big green point.
• The low-dimensional space found by FA is defined by W, which in this case
is a two-dimensional vector. Plot the low-dimensional space as a black line.
(Hint: This line should pass through µ.) Why is the low-dimensional space
found by FA different from that found by PCA and PPCA?
• Project each data point into the low-dimensional space using FA, and plot
each projected data point xˆn = W E[zn | xn] + µ as a red dot. (Hint: The
projected points should lie on the line defined by W.)
• Connect each data point xn with its projection xˆn using a red line