<img src="README/tud_logo.png" align="right" width="240"/>

# Piecewise monocular depth estimation by  plane fitting  

## Basic Overview

This paper proposes a modified approach for estimation dense depth estimation from monocular images. We model a complex 3D scene via over-segmentation via superpixels as a piecewise planar and rigid approximation. Based on this assumption we represent every planar by surface normals/plane coefficients. In this way we solve the homogeneous depth estimation problem that our baseline architecture Monodepth2 from [Godard et.al](https://github.com/nianticlabs/monodepth2)  2019 suffered. In particular we propose (i) a normal-2-block inside the architecture that estimates surface normal coefficients, (ii) a superpixel-loss that incorporates superpixel information and exploits sharper edges and (iii) a normal loss that ensure homogeneous depth for planar surfaces. We demonstrate the effectiveness of the proposed improvements in an detailed depth-map analysis and show comparable scoring metric with state-of-the-art results on the KITTI Eigen-Zhou split.

<p align="center"><img width=95% src="README/overview_architecture.png"></p>

## Results


## Abligation Studay & Comparison with other Networks

\begin{table*}[t]
    \begin{center}
    \resizebox{\textwidth}{!}{
    \begin{tabular}{|l||c|c|c|c||c|c|c|c|c|c|c|}
    \hline
         Method & Decoder & Ch & Sup & Loss & \cellcolor{red!25} Abs Rel & \cellcolor{red!25} Sq Rel & \cellcolor{red!25} RSME & \cellcolor{red!25} RSME\newline log & \cellcolor{blue!25}\delta<1.25 & \cellcolor{blue!25} \delta<1.25^{2} & \cellcolor{blue!25} \delta<1.25^{3} \\ \hline\hline
        Baseline & standard & 3 &  & standard & 0.115 & 0.903 & 4.863 & 0.193 & 0.877 & 0.959 & 0.981 \\ \hline
        N2D & normals  & 3 &  & standard & 0.123 & 0.984 & 5.042 & 0.2 & 0.859 & 0.955 & 0.98 \\ \hline
        4Ch & standard & 4 & fz & standard & 0.141 & 1.313 & 5.545 & 0.22 & 0.834 & 0.942 & 0.974 \\ \hline
        4Ch  & standard & 4 & sl & standard & 0.255 & 2.237 & 7.892 & 0.342 & 0.594 & 9.832 & 0.927 \\ \hline
        6Ch & standard & 6 & fz & standard & 0.122 & 0.978 & 5.026 & 0.2 & 0.862 & 0.955 & 0.979 \\ \hline
        4Ch + N2D & normals & 4 & fz & standard & 0.142 & 1.262 & 5.551 & 0.22 & 0.83 & 0.942 & 0.975 \\ \hline
        4Ch + N2D + bin & normals & 4 & fz & binary & 0.443 & 4.757 & 12.083 & 0.588 & 0.303 & 0.561 & 0.766 \\ \hline
        3Ch + N2D + bin & normals & 3 &  & binary & 0.443 & 4.757 & 12.083 & 0.588 & 0.303 & 0.561 & 0.766 \\ \hline
        4Ch + N2D + cont & normals & 4 & fz & continuous & 0.138 & 1.185 & 5.484 & 0.218 & 0.832 & 0.944 & 0.975 \\ \hline
        4Ch + N2D + cont & normals & 4 & sl & continuous & 0.443 & 4.757 & 12.083 & 0.588 & 0.303 & 0.561 & 0.766 \\ \hline
        3Ch + N2D + cont & normals & 3 &  & continuous & 0.443 & 4.757 & 12.083 & 0.588 & 0.303 & 0.561 & 0.766 \\ \hline
        4Ch + N2D +  0.001 norm & normals & 4 & fz & 0.001 * norm & 0.139 & 1.193 & 5.525 & 0.22 & 0.831 & 0.941 & 0.974 \\ \hline
        4Ch + N2D +  0.01 norm & normals & 4 & fz & 0.01 * norm & 0.139 & 1.172 & 5.513 & 0.218 & 0.831 & 0.942 & 0.975 \\ \hline
        4Ch + N2D +  0.1 norm & normals & 4 & fz & 0.1 * norm & 0.443 & 4.757 & 12.083 & 0.588 & 0.303 & 0.561 & 0.766 \\ \hline
        4Ch + N2D + bin + norm & normals & 4 & fz & bin + norm & 0.443 & 4.757 & 12.083 & 0.588 & 0.303 & 0.561 & 0.766 \\ \hline
        4Ch + N2D + cont + norm & normals & 4 & fz & cont + norm & 0.141 & 1.276 & 5.549 & 0.221 & 0.832 & 0.941 & 0.974 \\ \hline
        4Ch + N2D + cont + norm & normals & 4 & sl & cont + norm & 0.443 & 4.757 & 12.083 & 0.588 & 0.303 & 0.561 & 0.766 \\ \hline
        3Ch + N2D + cont + norm & normals & 3 &  & cont + norm & 0.443 & 4.757 & 12.083 & 0.588 & 0.303 & 0.561 & 0.766 \\ \hline
    \end{tabular}}
    \end{center}
    \caption{\textbf{Ablation.} Results for different variants of our model with monocular training on KTTI \cite{KITTI13} test set. All variants trained with the KITTI Eigen-Zhou split \cite{Zhou2017}. The methods using our proposed normal-to-depth block are denoted by N2D (normals decoder). The number of input channels (Ch) have been modified to 3, 4 or 6. The method to calculate the superpixel used by the variant is denoted by fz for Felzenwalb's method \cite{Felzen2004} or by sl for SLIC \cite{slic2010}.}
    
\end{table*}
