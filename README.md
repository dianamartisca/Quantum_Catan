# Quantum Catan 🧩

Quantum Catan is a collection of algorithms and tools for solving and simulating problems inspired by the board game Catan, using quantum computing techniques such as QAOA (Quantum Approximate Optimization Algorithm).


## Project Structure and Task Descriptions

🏠 **task1/**: **Quantum Settlement Planner**
   - Optimal resource placement using quantum optimization (QAOA/VQE).
   - Place two or more settlements on a hex board to maximize expected resource yield.
   - Constraints: settlements must be at least 2 edges apart; objective is to maximize the weighted sum of resource values accessible to each settlement.
   - Includes quantum encoding, QAOA/VQE implementation, and visualization of optimal placements.

🛣️ **task2/**: **The Quantum Longest Road**
   - Path finding and connectivity optimization using QAOA/Quantum Annealing.
   - Find the longest connected path you can build on a reduced Catan road network, given resource constraints and blocked edges.
   - Objective: maximize the length of the connected road while respecting limited resources and rival blockages.

🌾 **task3/**: **Quantum Resource Trader**
   - Resource optimization under constraints (QAOA/Grover/Hybrid Classical-Quantum).
   - Decide which actions (build/trade) to execute to maximize your score, given an inventory of resources and possible actions.
   - Modeled as a quantum knapsack: each action is a binary variable, each consumes resources and gives points, with the objective to maximize total points without exceeding resource limits.

📝 **requirements.txt**: Python dependencies for the project.


## Getting Started 🚀

1. **Clone the repository**
   ```sh
   git clone <repo-url>
   cd Quantum_Catan
   ```
2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```
3. **Explore the tasks**
   - Each `taskX/` folder contains Python modules for a specific quantum optimization challenge inspired by Catan.
   - See the main algorithm files in each folder for entry points and usage (e.g., `qaoa.py`, `qaoa_max_cut.py`).

## Requirements
- Python 3.8+
- See `requirements.txt` for required packages (e.g., Qiskit, NumPy, etc.)

## Usage
- Run individual scripts in each task folder to execute specific algorithms or simulations.
- Example:
  ```sh
  python task1/qaoa.py
  ```
