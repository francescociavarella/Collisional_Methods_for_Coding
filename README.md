# Collisional_Methods_for_Coding
Simulation of **Excitonic transport** dynamics in a **excitonic dimer** using a stochastic unraveling approach based on **Collisional Methods**, both in **Quantum Jump** and **Difussive Limits**.
The results are validated through a comparison with the dynamics derived from the **Lindblad Master Equation**. Furthermore, the collisional evolution is benchmarked against the **Isolated System** dynamics (i.e., purely unitary evolution without the collisional interaction).

Author : Francesco Ciavarella - francesco.ciavarella@studenti.unipd.it

---
## Contents : 

1. [Initial Configuration](#1-Initial-Configuration)
2. [Repository Organization](#2-Repository-Organization)
3. [Future Goals](#3-Future-Goals)

---

## Initial Configuration
To replicate the results clone the repository and recreate the Conda enviroment, to ensure all required libraries are installed.

1. **Clone the Repository**
    ```
    git clone https://github.com/francescociavarella/Collisional_Methods_for_Coding.git
   cd Collisional_Methods_for_Coding
    ```
2. **Create the Enviroment wit Conda**
    ```
    conda env create -f enviroment.yml
    ```
3. **Activate Enviroment**
    ```
    conda avtivate mc
    ```
Now you are ready to navigate the repository and run the notebooks.

---

## Repository Organization

This repository is organized in three main sections:

* **Codes** : contains the `Main_Dynamics` program for implementation of the system evolution;it also includes two notebook for visualization of the results: `Plot.ipynb` for standard plotting and `Bloch_Sphere.ipynb` for Bloch's Sphere animation .

* **Results** : contains some example of simulation results. 

* **Documentation** : contains `Report.pdf` which provides a brief introduction to *Collisional Methods* and the physical basis of the main program. Additionally `Demonstration.pdf` describes the mathematical derivation of some of the most important equations used. 

---

## Future Goals

Currently the focus is on the investigation and validation of an hypothetical **Intermediate Limit**, between the two regimes already analized (i.e. Quantum Jump and Diffusive); the goal is characterized the resulting unraveling and its physical dependencies.

Future work will be the investigation of a **more complex dynamics and system topology**, for example using more than two site or introducing other different Enviroment effects (beyond the pure Dephasing currently studied)

---

