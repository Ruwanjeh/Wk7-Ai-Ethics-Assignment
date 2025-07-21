# Bonus Task: Simple Quantum Circuit with Qiskit
# This script demonstrates a basic quantum circuit and explains its relevance to AI optimization tasks.

from qiskit import QuantumCircuit, Aer, execute
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# Create a quantum circuit with 2 qubits and 2 classical bits
qc = QuantumCircuit(2, 2)

# Apply Hadamard gates to put both qubits in superposition
qc.h(0)
qc.h(1)

# Measure the qubits
qc.measure([0, 1], [0, 1])

# Simulate the circuit
simulator = Aer.get_backend('qasm_simulator')
result = execute(qc, simulator, shots=1024).result()
counts = result.get_counts(qc)

# Plot the result
print("Measurement results:", counts)
plot_histogram(counts)
plt.show()

# Explanation:
# Quantum circuits like this are the foundation for quantum algorithms (e.g., QAOA, Grover's search)
# that can solve optimization problems much faster than classical computers for certain tasks.
# In AI, this could accelerate drug discovery by efficiently searching large chemical spaces. 