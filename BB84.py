import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import random

class BB84Protocol:
    def __init__(self, key_length=8):
        """
        Initialize BB84 protocol simulation
        
        Args:
            key_length (int): Number of bits in the key
        """
        self.key_length = key_length
        self.alice_bits = []
        self.alice_bases = []
        self.bob_bases = []
        self.bob_results = []
        self.shared_key = []
        self.simulator = AerSimulator()
        self.detect_array = []
        self.detect_evesdropping = False
    def generate_random_bits(self):
        """Generate random bits and bases for Alice"""
        self.alice_bits = [random.randint(0, 1) for _ in range(self.key_length)]
        self.alice_bases = [random.randint(0, 1) for _ in range(self.key_length)]
        # print(f"Alice's random bits: {self.alice_bits}")
        # print(f"Alice's random bases: {self.alice_bases} (0=Z, 1=X)")
        
    def alice_prepare_qubits(self):
        """Alice prepares qubits based on her bits and bases"""
        self.circuits = []
        
        for i in range(self.key_length):
            # Create quantum circuit for each qubit
            qc = QuantumCircuit(1, 1)
            
            # Prepare qubit based on bit value and basis choice
            if self.alice_bits[i] == 1:
                qc.x(0)  # Apply X gate if bit is 1
                
            if self.alice_bases[i] == 1:
                qc.h(0)  # Apply Hadamard if using X basis
                
            self.circuits.append(qc)
            
        print("\nAlice prepared qubits and sent them to Bob")
        
    def bob_measure_qubits(self):
        """Bob randomly chooses bases and measures qubits"""
        self.bob_bases = [random.randint(0, 1) for _ in range(self.key_length)]
        self.bob_results = []
        
        # print(f"Bob's measurement bases: {self.bob_bases} (0=Z, 1=X)")
        
        for i in range(self.key_length):
            # Create a copy of Alice's circuit
            qc = self.circuits[i].copy()
            
            # Bob applies his measurement basis
            if self.bob_bases[i] == 1:
                qc.h(0)  # Apply Hadamard if measuring in X basis
                
            # Add measurement
            qc.measure(0, 0)
            
            # Execute circuit
            job = self.simulator.run(qc, shots=1)
            result = job.result()
            counts = result.get_counts()
            
            # Get measurement result
            measured_bit = int(list(counts.keys())[0])
            self.bob_results.append(measured_bit)
            
        # print(f"Bob's measurement results: {self.bob_results}")
        
    def sift_key(self):
        """Compare bases and keep bits where bases match"""
        self.shared_key = []
        matching_indices = []
        
        for i in range(self.key_length):
            if self.alice_bases[i] == self.bob_bases[i]:
                self.shared_key.append(self.alice_bits[i])
                matching_indices.append(i)
        
        # print(f"\nMatching basis indices: {matching_indices}")
        # print(f"Sifted key length: {len(self.shared_key)}")
        # print(f"Alice's sifted key: {self.shared_key}")
        
        # Verify Bob's corresponding bits
        bob_sifted = [self.bob_results[i] for i in matching_indices]
        # print(f"Bob's sifted key:   {bob_sifted}")
        
        # Check for errors (in ideal case, should be identical)
        errors = sum(1 for a, b in zip(self.shared_key, bob_sifted) if a != b)
        error_rate = errors / len(self.shared_key) if self.shared_key else 0

        ## check which is key
        indices = []
        for i in range(len(self.check_success_basis)//2+1):
            indices.append(self.check_success_basis[i])
        key = [self.bob_results[i] for i in indices]
        key_in_alice = [self.alice_bits[i] for i in indices]

        print(f"Bit error rate: {error_rate:.2%}")
        # print(f"Key from Bob: {key}")
        # print(f"Key from Alice: {key_in_alice}")
        print(f"Final_key is same ? {key == key_in_alice}")
        self.shared_key = key
        return self.shared_key
    def check_evesdropping(self):
        for i in range(len(self.check_success_basis)//2):
            if self.alice_bits[self.check_success_basis[i]] != self.bob_results[self.check_success_basis[i]]:
                self.detect_evesdropping = True
        
                    # print(f"Error detected at index {i}: Alice's bit {self.alice_bits[i]} != Bob's bit {self.bob_results[i]}")
        print(f"Is there eavesdropping? {self.detect_evesdropping}")
    def run_protocol(self):
        """Execute the complete BB84 protocol"""
        print("=" * 50)
        print("BB84 QUANTUM KEY DISTRIBUTION SIMULATION")
        print("=" * 50)
        
        print("\n1. Alice generates random bits and bases")
        self.generate_random_bits()
        
        print("\n2. Alice prepares and sends qubits")
        self.alice_prepare_qubits()
        
        print("\n3. Bob measures qubits with random bases")
        self.bob_measure_qubits()
        self.check_success_basis = [i for i in range(self.key_length) if self.alice_bases[i] == self.bob_bases[i]]
        print("\n3.1. Check for eavesdropping")
        self.check_evesdropping()
        print("\n4. Basis comparison and key sifting")
        final_key = self.sift_key()
        
        # print(f"\n5. Final shared secret key: {final_key}")
        print(f"Key efficiency: {len(final_key)}/{self.key_length} = {len(final_key)/self.key_length:.1%}")
        
        return final_key
        
    def demonstrate_eavesdropping(self):
        """Demonstrate how eavesdropping introduces errors"""
        print("\n" + "=" * 50)
        print("EAVESDROPPING DETECTION DEMONSTRATION")
        print("=" * 50)
        
        # Reset for eavesdropping simulation
        self.generate_random_bits()
        
        # Eve intercepts and measures
        eve_bases = [random.randint(0, 1) for _ in range(self.key_length)]
        # print(f"Eve's interception bases: {eve_bases}")
        
        # Alice prepares qubits
        self.circuits = []
        for i in range(self.key_length):
            ## qc 1 qbit, 1 classical bit
            qc = QuantumCircuit(1, 1)
            
            if self.alice_bits[i] == 1:
                qc.x(0) ##apply on 0th bit
            if self.alice_bases[i] == 1:
                qc.h(0)
                
            # Eve measures and retransmits
            if eve_bases[i] == 1:
                qc.h(0)
            # print(qc)
            #
            qc.measure(0, 0) #measure the qubit and store result in classical bit 0th
            # print(qc)
            # Simulate Eve's measurement
            temp_qc = qc.copy()
            job = self.simulator.run(temp_qc, shots=1)
            result = job.result()
            eve_bit = int(list(result.get_counts().keys())[0])
            
            # Eve prepares new qubit based on her measurement
            qc_new = QuantumCircuit(1, 1)
            if eve_bit == 1:
                qc_new.x(0)
            if eve_bases[i] == 1:
                qc_new.h(0)
                
            self.circuits.append(qc_new)
            
        # Bob measures normally
        self.bob_measure_qubits()
        self.check_evesdropping()
        self.sift_key()
        
        print("Notice the increased error rate due to eavesdropping!")

# Example usage and demonstration
def main():
    # Basic BB84 protocol
    bb84 = BB84Protocol(key_length=500)
    shared_key = bb84.run_protocol()
    
    # Demonstrate eavesdropping detection
    bb84.demonstrate_eavesdropping()
    
    print("\n" + "=" * 50)
    print("PROTOCOL SUMMARY")
    print("=" * 50)
    print("The BB84 protocol allows Alice and Bob to generate a shared secret key")
    print("using quantum mechanics. Any eavesdropping attempt will introduce")
    print("detectable errors due to the quantum no-cloning theorem.")

if __name__ == "__main__":
    main()