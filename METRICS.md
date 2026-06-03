# Metric Definitions

This file defines the main metrics used in the PR-DDOLF manuscript and in the
released source code.

## Notation

For a sample, let \(y(T, \omega)\) denote the experimental response and
\(\hat{y}(T, \omega)\) denote the predicted response at temperature \(T\) and
frequency \(\omega\). The response can be storage modulus \(E'\), loss modulus
\(E''\), or loss factor \(\tan\delta\).

The processed `.npz` files expected by the scripts store:

- \(E'\) and \(E''\) as raw modulus values in Pa;
- \(\tan\delta\) as a dimensionless value;
- temperature in degrees Celsius;
- frequency in Hz.

Model code may evaluate modulus errors in log space. Manuscript figures may
display modulus values in MPa after unit conversion from Pa. Manuscript plots
may report RMSE values as percentages after multiplying the stored decimal
metric by 100.

## Measured-range RMSE

For \(E'\) and \(E''\), the measured-range RMSE is computed on the valid measured
support region of the corresponding experimental curve. In the released code,
the primary modulus reconstruction error is evaluated in log space:

\[
\mathrm{RMSE}_{\log}(y,\hat{y}) =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
\left[
\log_{10}(\hat{y}_i+\varepsilon) -
\log_{10}(y_i+\varepsilon)
\right]^2
},
\]

where \(N\) is the number of valid measured points and \(\varepsilon\) is a small
positive constant used to avoid taking the logarithm of zero. When reported as a
percentage in manuscript figures, the decimal value is multiplied by 100.

## Loss-factor error

For \(\tan\delta\), the preferred curve-level error is the mean absolute error:

\[
\mathrm{MAE}_{\tan\delta} =
\frac{1}{N}\sum_{i=1}^{N}
|\hat{y}_i-y_i|.
\]

If \(\tan\delta\) is shown as a heat map or temperature-wise curve, the plotted
quantity is the pointwise absolute error:

\[
e_{\tan\delta}(T_i)=|\hat{y}(T_i)-y(T_i)|.
\]

## Relative error curves

For visualization of pointwise relative error in modulus responses, the relative
error is:

\[
e_{\mathrm{rel}}(T_i) =
\frac{|\hat{y}(T_i)-y(T_i)|}
{\max(|y(T_i)|,\varepsilon)}
\times 100\%.
\]

This pointwise relative error is used for visual diagnostics only and should not
be confused with the log-space RMSE used in the main modulus summary metrics.

## Descriptor CV

For a constitutive descriptor \(\theta_j\) inferred from multiple anchor or
auxiliary frequency views of the same sample, the descriptor coefficient of
variation is:

\[
\mathrm{CV}_j =
\frac{\mathrm{std}(\theta_j)}
{\left|\mathrm{mean}(\theta_j)\right|+\varepsilon}.
\]

The reported descriptor CV is the average over the descriptor dimensions used in
the corresponding evaluation script:

\[
\mathrm{Descriptor\ CV} =
\frac{1}{D}\sum_{j=1}^{D}\mathrm{CV}_j.
\]

When shown as a percentage, this value is multiplied by 100.

## Descriptor relative L2

For two descriptor vectors inferred from two frequency views of the same sample,
\(\boldsymbol{\theta}^{(a)}\) and \(\boldsymbol{\theta}^{(b)}\), the relative
L2 difference is:

\[
\mathrm{RelL2}(a,b)=
\frac{
\left\|\boldsymbol{\theta}^{(a)}-\boldsymbol{\theta}^{(b)}\right\|_2
}{
\left\|\boldsymbol{\theta}^{(a)}\right\|_2+\varepsilon
}.
\]

The reported descriptor relative L2 is the average over the evaluated
frequency-view pairs and test samples. When shown as a percentage, this value is
multiplied by 100.

## Practical usability score

The practical usability score is an aggregate descriptor-readiness score used to
summarize whether first-stage descriptor identification is sufficiently stable
for downstream residual-field correction. It combines descriptor consistency,
descriptor dispersion, and reconstruction-quality criteria. Higher values mean
better practical usability.

If a strict ready ratio is used, a sample is counted as ready only when all
predefined criteria are satisfied:

\[
\mathrm{Ready\ ratio} =
\frac{\#\{\mathrm{ready\ test\ samples}\}}
{\#\{\mathrm{test\ samples}\}}.
\]

If the continuous scoring version is used, each criterion contributes partial
credit and the final score is reported as a percentage. The exact thresholds and
weights should be taken from the evaluation script used for the corresponding
figure.
