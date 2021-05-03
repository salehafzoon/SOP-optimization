# SOP optimization

Solving SOP with SA, GRASP, Tabu Search algorithms.

> python implemented 
[![Python 3.9.4](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

## Problem Description
The Sequential Ordering Problem **(SOP)** with precedence constraints consists of finding a minimum weight Hamiltonian path on a directed graph with weights on the arcs and on the   nodes, subject to precedence constraints among nodes.

<div align="center">
  <img src="./photos/SOP-img1.PNG" width="310px">
  <img src="./photos/SOP-img2.PNG" width="290px">
</div>



## Instance Description
**TSPLIB** is a library of sample instanes for the TSP (and related problems) from various soures and of various types.
Instanes of the following problem classes are available:
+ Symmetric traveling salesman problem **(TSP)**
+ Hamiltonian cycle problem **(HCP)**
+ Asymmetric traveling salesman problem **(ATSP)**
+ Sequential ordering problem **(SOP)**
+ Capaitated vehicle routing problem **(CVRP)**
- ### The file format
  Each file consists of **specification part** and of a **data part**. The specification part contains information on the file format and on its contents. The data part contains explicit data. 
python implementation

## Algorithms

All the algorithms are designed base on the **Lexicographic Search** using **forwarding and back warding path-preserving-3-exchange** as explained in [An Ant Colony System Hybridized with a New Local Search for the Sequential Ordering Problem paper](http://people.idsia.ch/~luca/fd18cc00a0_article.pdf).

<div align="center">
  <img src="./photos/3-exchange.png" width="68%">
</div>

## Running
Just go to related algorithm directory and run the python code. example:
```
 cd ./SA
 python main.py
```

## Results
All results and comparing analyze are presented in **Report.pdf** files.
overal algorithms comparisons result is as follow:
<div align="center">
  <img src="./photos/pic3.PNG" width="70%">
</div>

## References
- An Ant Colony System Hybridized with a New Local Search for the Sequential Ordering Problem paper [Link to the paper](http://people.idsia.ch/~luca/fd18cc00a0_article.pdf).
- SOP images srouce => The paper : [Tree search algorithms for the Sequential Ordering Problem](https://arxiv.org/pdf/1911.12427.pdf)
