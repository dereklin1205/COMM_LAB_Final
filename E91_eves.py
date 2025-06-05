import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import random

class E91ProtocolWithEavesdropping:
    def __init__(self, key_length=200, eavesdropping_probability=0.0):
        """
        Initialize E91 protocol simulation with eavesdropping capability
        
        Args:
            key_length (int): Number of bits in the key
            eavesdropping_probability (float): Probability that Eve intercepts each photon (0.0 to 1.0)
        """
        self.key_length = key_length
        self.eavesdropping_probability = eavesdropping_probability
        self.basesAsja = []
        self.basesBalvis = []
        self.keyAsja = []
        self.keyBalvis = []
        self.final_key_Asja = []
        self.final_key_Balvis = []
        self.diffAsja = []
        self.diffAsja_bases = []
        self.diffBalvis = []
        self.diffBalvis_bases = []
        self.eve_detected = False
        
    def run_protocol(self):
        """
        Run the E91 protocol to generate a shared key
        """
        print(f"Running E91 protocol with eavesdropping probability: {self.eavesdropping_probability}")
        self.gen_key_and_measure(self.key_length)
        self.compare_bases()
        chsh_value = self.compute_CHSH()
        print("E91 protocol completed.")
        print(f"Final key is same for Asja and Balvis? {self.final_key_Asja == self.final_key_Balvis}")
        return chsh_value
        
    def compare_bases(self):
        for i in range(self.key_length):
            if self.basesAsja[i] == self.basesBalvis[i]:
                self.final_key_Asja.append(self.keyAsja[i])
                self.final_key_Balvis.append(self.keyBalvis[i])
            else:
                self.diffAsja.append(self.keyAsja[i])
                self.diffAsja_bases.append(self.basesAsja[i])
                self.diffBalvis.append(self.keyBalvis[i])
                self.diffBalvis_bases.append(self.basesBalvis[i])
        
        print(f"Matching bases found: {len(self.final_key_Asja)} out of {self.key_length}")
        print(f"Key agreement rate: {len(self.final_key_Asja)/self.key_length:.2%}")
        
    def compute_CHSH(self):
        """
        Compute CHSH inequality value to detect eavesdropping
        """
        # ZW correlation
        sameZW, diffZW = 0, 0
        for i, (baseA, baseB) in enumerate(zip(self.diffAsja_bases, self.diffBalvis_bases)):
            if baseA == 'Z' and baseB == 'W':
                if self.diffAsja[i] == self.diffBalvis[i]:
                    sameZW += 1
                else:
                    diffZW += 1
        totalZW = sameZW + diffZW
        ZW = (sameZW - diffZW) / totalZW if totalZW != 0 else 0
        
        # ZV correlation
        sameZV, diffZV = 0, 0
        for i, (baseA, baseB) in enumerate(zip(self.diffAsja_bases, self.diffBalvis_bases)):
            if baseA == 'Z' and baseB == 'V':
                if self.diffAsja[i] == self.diffBalvis[i]:
                    sameZV += 1
                else:
                    diffZV += 1
        totalZV = sameZV + diffZV
        ZV = (sameZV - diffZV) / totalZV if totalZV != 0 else 0
        
        # XW correlation
        sameXW, diffXW = 0, 0
        for i, (baseA, baseB) in enumerate(zip(self.diffAsja_bases, self.diffBalvis_bases)):
            if baseA == 'X' and baseB == 'W':
                if self.diffAsja[i] == self.diffBalvis[i]:
                    sameXW += 1
                else:
                    diffXW += 1
        totalXW = sameXW + diffXW
        XW = (sameXW - diffXW) / totalXW if totalXW != 0 else 0
        
        # XV correlation
        sameXV, diffXV = 0, 0
        for i, (baseA, baseB) in enumerate(zip(self.diffAsja_bases, self.diffBalvis_bases)):
            if baseA == 'X' and baseB == 'V':
                if self.diffAsja[i] == self.diffBalvis[i]:
                    sameXV += 1
                else:
                    diffXV += 1
        totalXV = sameXV + diffXV
        XV = (sameXV - diffXV) / totalXV if totalXV != 0 else 0
        
        # CHSH value
        S = abs(ZW + ZV + XW - XV)
        
        print(f"\nCHSH Analysis:")
        print(f"ZW correlation: {ZW:.3f} (samples: {totalZW})")
        print(f"ZV correlation: {ZV:.3f} (samples: {totalZV})")
        print(f"XW correlation: {XW:.3f} (samples: {totalXW})")
        print(f"XV correlation: {XV:.3f} (samples: {totalXV})")
        print(f"CHSH value (S): {S:.3f}")
        
        # In quantum mechanics without eavesdropping, S can be up to 2√2 ≈ 2.83
        # Classical limit is S ≤ 2
        # With eavesdropping, S typically approaches 2 or below
        if S > 2.5:
            print("✓ Strong quantum correlation detected - No eavesdropping")
            self.eve_detected = False
        elif S > 2.2:
            print("⚠ Moderate quantum correlation - Possible light eavesdropping")
            self.eve_detected = False
        else:
            print("⚠ Weak correlation detected - POTENTIAL EAVESDROPPING!")
            self.eve_detected = True
            
        return S
    
    def gen_key_and_measure(self, key_length=200):
        """
        Generate entangled pairs and perform measurements with potential eavesdropping
        """
        for i in range(key_length):
            qc = QuantumCircuit(3, 3)  # Extra qubit for Eve's measurement
            
            # Creating entanglement between qubits 0 and 1
            qc.h(0)
            qc.cx(0, 1)
            
            # Eve's eavesdropping attempt
            if random.random() < self.eavesdropping_probability:
                # Eve intercepts qubit 1 and measures it
                # She then prepares a new qubit based on her measurement
                # This breaks the entanglement and introduces errors
                
                # Eve measures qubit 1 in a random basis
                eve_basis = random.randint(0, 2)
                if eve_basis == 1:  # X basis
                    qc.h(1)
                elif eve_basis == 2:  # Diagonal basis
                    qc.s(1)
                    qc.h(1)
                
                # Measure Eve's qubit
                qc.measure(1, 2)
                
                # Eve prepares a new qubit for Balvis based on her measurement
                # This is a simplified model - in reality this is more complex
                qc.reset(1)
                if eve_basis == 1:  # If Eve measured in X, prepare accordingly
                    qc.h(1)
                elif eve_basis == 2:  # If Eve measured in diagonal
                    qc.h(1)
                    qc.s(1)
            
            # Asja chooses measurement basis
            choiceAsja = random.randint(0, 2)
            choiceBalvis = random.randint(0, 2)
            
            if choiceAsja == 0:  # Z basis
                self.basesAsja.append('Z')
            elif choiceAsja == 1:  # X basis
                qc.h(0)
                self.basesAsja.append('X')
            elif choiceAsja == 2:  # W basis
                qc.s(0)
                qc.h(0)
                qc.t(0)
                qc.h(0)
                self.basesAsja.append('W')
            
            # Balvis chooses measurement basis
            if choiceBalvis == 0:  # Z basis
                self.basesBalvis.append('Z')
            elif choiceBalvis == 1:  # W basis
                qc.s(1)
                qc.h(1)
                qc.t(1)
                qc.h(1)
                self.basesBalvis.append('W')
            elif choiceBalvis == 2:  # V basis
                qc.s(1)
                qc.h(1)
                qc.tdg(1)
                qc.h(1)
                self.basesBalvis.append('V')
            
            # Measurement
            qc.measure([0, 1], [0, 1])
            
            # Simulate the circuit
            simulator = AerSimulator()
            job = simulator.run(qc, shots=1)
            result = job.result()
            counts = result.get_counts(qc)
            
            # Extract measurement results
            measurement_result = list(counts.keys())[0]
            self.keyAsja.append(int(measurement_result[-1]))  # Last bit for Asja
            self.keyBalvis.append(int(measurement_result[-2]))  # Second to last for Balvis

def compare_eavesdropping_levels(key_length = 1000):
    """
    Compare CHSH values across different eavesdropping levels
    """
    print("=" * 60)
    print("EAVESDROPPING DETECTION DEMONSTRATION")
    print("=" * 60)
    
    eavesdrop_levels = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0]
    chsh_values = []
    
    for prob in eavesdrop_levels:
        print(f"\n--- Testing with {prob:.0%} eavesdropping probability ---")
        protocol = E91ProtocolWithEavesdropping(key_length, eavesdropping_probability=prob)
        chsh_value = protocol.run_protocol()
        chsh_values.append(chsh_value)
    
    print("\n" + "=" * 60)
    print("SUMMARY OF RESULTS")
    print("=" * 60)
    print("Eavesdrop%  | CHSH Value | Detection")
    print("-" * 40)
    for prob, chsh in zip(eavesdrop_levels, chsh_values):
        detection = "None" if chsh > 2.5 else "Light" if chsh > 2.2 else "DETECTED"
        print(f"{prob:8.0%}   | {chsh:8.3f}   | {detection}")
    
    print(f"\nKey insight: As eavesdropping increases, CHSH value decreases,")
    print(f"making detection possible. Perfect eavesdropping (100%) typically")
    print(f"results in CHSH ≤ 2, clearly indicating security breach.")

if __name__ == "__main__":
    # Run the comparison to show how eavesdropping affects the protocol
    compare_eavesdropping_levels(key_length=10000)
    
    print(f"\n" + "=" * 60)
    print("EDUCATIONAL NOTE:")
    print("This simulation demonstrates how quantum cryptography detects")
    print("eavesdropping through violation of Bell inequalities (CHSH test).")
    print("In real implementations, this provides provable security.")
    print("=" * 60)