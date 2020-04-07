<img src="README/tud_logo.png" align="right" width="240"/>

# Piecewise monocular depth estimation by  plane fitting  

## Basic Overview

This paper proposes a modified approach for estimation dense depth estimation from monocular images. We model a complex 3D scene via over-segmentation via superpixels as a piecewise planar and rigid approximation. Based on this assumption we represent every planar by surface normals/plane coefficients. In this way we solve the homogeneous depth estimation problem that our baseline architecture Monodepth2 from [Godard et.al](https://github.com/nianticlabs/monodepth2)  2019 suffered. In particular we propose (i) a normal-2-block inside the architecture that estimates surface normal coefficients, (ii) a superpixel-loss that incorporates superpixel information and exploits sharper edges and (iii) a normal loss that ensure homogeneous depth for planar surfaces. We demonstrate the effectiveness of the proposed improvements in an detailed depth-map analysis and show comparable scoring metric with state-of-the-art results on the KITTI Eigen-Zhou split.

<p align="center"><img width=95% src="README/toverview_architecture.png"></p>

## Results


## 