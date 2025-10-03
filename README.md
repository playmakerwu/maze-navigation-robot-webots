# Maze Navigation Robot with Webots  
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)  
[![Simulator](https://img.shields.io/badge/Simulator-Webots-green.svg)](https://cyberbotics.com/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  

A maze navigation robot built in **Webots**, featuring **Particle Filter (Monte Carlo Localization)** for robot localization and **global A\*** path planning with **differential-drive control**.  

---

## 📌 Overview  
This project demonstrates a fully integrated autonomous robot system in simulation. The robot is able to:  
- Localize itself in a maze-like environment using **Particle Filter** with noisy odometry and landmark observations.  
- Plan a global path to a target using the **A\*** search algorithm.  
- Execute the path with **differential-drive control**.  
- Use a **finite-state machine** (FSM) controller for higher-level behaviors such as searching, aligning, turning, and pushing a ball toward a goal.  

---

## 🧩 System Architecture  


---

## 🚀 Features  
- **Particle Filter Localization**:  
  - Predict → Update → Resample → Estimate pose.  
  - Robust handling of noisy odometry and sensor data.  

- **Global Path Planning**:  
  - A* algorithm for optimal path search.  
  - Generates waypoints from start to target.  

- **Differential Drive Control**:  
  - Wheel velocity commands calculated from position/angle errors.  
  - Navigation with proportional control gains.  

- **Finite-State Machine Controller**:  
  - `SEARCHING_FOR_BALL`: Rotate to detect the ball.  
  - `CALCULATING_ALIGN_TARGETS`: Compute alignment point behind the ball.  
  - `NAVIGATING_TO_ALIGN_POINT`: Move to the alignment target.  
  - `TURNING_TOWARDS_BALL_POS`: Rotate to face the ball.  
  - `PUSHING_BALL`: Push the ball toward the goal.  

---

## 📂 Project Structure  


---

## ⚙️ Installation & Usage  

### Prerequisites  
- [Webots](https://cyberbotics.com/) (R2023a or newer recommended)  
- Python 3.8+  
- Required Python libraries:  
  ```bash
  pip install numpy

git clone https://github.com/playmakerwu/maze-navigation-robot-webots.git
cd maze-navigation-robot-webots
