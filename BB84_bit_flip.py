import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import random

class BB84Protocol:
    def __init__(self, key_length=8, bit_flip_probability=0.0):
        """
        Initialize BB84 protocol simulation with bit flip error
        
        Args:
            key_length (int): Number of bits in the key
            bit_flip_probability (float): Probability of bit flip error (0.0 to 1.0)
        """
        self.key_length = key_length
        self.bit_flip_probability = bit_flip_probability
        self.alice_bits = []
        self.alice_bases = []
        self.bob_bases = []
        self.bob_results = []
        self.shared_key = []
        self.simulator = AerSimulator()
        self.detect_array = []
        self.detect_evesdropping = False
        self.bit_flip_locations = []
        
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
        
    def apply_bit_flip_error(self):
        """Apply bit flip errors during transmission"""
        self.bit_flip_locations = []
        
        for i in range(self.key_length):
            if random.random() < self.bit_flip_probability:
                # Apply bit flip (X gate) to the qubit
                self.circuits[i].x(0)
                self.bit_flip_locations.append(i)
                
        # if self.bit_flip_locations:
        #     print(f"Bit flip errors occurred at positions: {self.bit_flip_locations}")
        # else:
        #     print("No bit flip errors occurred during transmission")
        
    def bob_measure_qubits(self):
        """Bob randomly chooses bases and measures qubits"""
        self.bob_bases = [random.randint(0, 1) for _ in range(self.key_length)]
        self.bob_results = []
        
        # print(f"Bob's measurement bases: {self.bob_bases} (0=Z, 1=X)")
        
        for i in range(self.key_length):
            # Create a copy of Alice's circuit (now potentially with bit flip errors)
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
        
    def analyze_bit_flip_effects(self):
        """Analyze the effects of bit flip errors on different measurement bases"""
        print("\n" + "=" * 50)
        print("BIT FLIP ERROR ANALYSIS")
        print("=" * 50)
        
        if not self.bit_flip_locations:
            print("No bit flip errors to analyze")
            return
            
        print("Effects of bit flip errors:")
        print("- Bit flip (X gate) DOES affect measurements in Z basis (computational basis)")
        print("- Bit flip has NO effect when measuring in X basis (Hadamard basis)")
        print()
        
        for pos in self.bit_flip_locations:
            alice_bit = self.alice_bits[pos]
            alice_basis = self.alice_bases[pos]
            bob_basis = self.bob_bases[pos]
            bob_result = self.bob_results[pos]
            
            print(f"Position {pos}:")
            print(f"  Alice: bit={alice_bit}, basis={'Z' if alice_basis==0 else 'X'}")
            print(f"  Bob:   basis={'Z' if bob_basis==0 else 'X'}, result={bob_result}")
            
            if alice_basis == bob_basis:  # Matching bases
                if alice_basis == 0:  # Z basis
                    print(f"  → Z basis measurement: Bit flip FLIPS the result")
                    expected_without_error = alice_bit
                    print(f"  → Expected without error: {expected_without_error}, Got: {bob_result}, Flipped: {expected_without_error != bob_result}")
                else:  # X basis
                    print(f"  → X basis measurement: Bit flip has NO effect")
                    print(f"  → Expected: {alice_bit}, Got: {bob_result}, Match: {alice_bit == bob_result}")
            else:
                print(f"  → Different bases: Result is random anyway")
            print()
    
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
        
        # Check for errors (should show bit flip errors in Z basis measurements)
        errors = sum(1 for a, b in zip(self.shared_key, bob_sifted) if a != b)
        error_rate = errors / len(self.shared_key) if self.shared_key else 0
        
        # Analyze which errors are due to bit flips
        bit_flip_errors = 0
        for i, idx in enumerate(matching_indices):
            if idx in self.bit_flip_locations:
                if self.alice_bases[idx] == 0:  # Z basis measurement
                    bit_flip_errors += 1
                    # print(f"  Bit flip error detected at position {idx} (Z basis)")
        
        print(f"\nBit error rate: {error_rate:.2%}")
        print(f"Errors due to bit flips in Z basis: {bit_flip_errors}")
        print(f"Other errors: {errors - bit_flip_errors}")
        
        # For demonstration, use Alice's key as the reference
        return self.shared_key
    
    def run_protocol(self):
        """Execute the complete BB84 protocol with bit flip errors"""
        print("=" * 50)
        print("BB84 QUANTUM KEY DISTRIBUTION WITH BIT FLIP ERRORS")
        print("=" * 50)
        
        print(f"Bit flip probability: {self.bit_flip_probability:.2%}")
        
        print("\n1. Alice generates random bits and bases")
        self.generate_random_bits()
        
        print("\n2. Alice prepares and sends qubits")
        self.alice_prepare_qubits()
        
        print("\n3. Apply bit flip errors during transmission")
        self.apply_bit_flip_error()
        
        print("\n4. Bob measures qubits with random bases")
        self.bob_measure_qubits()
        
        print("\n5. Analyze bit flip effects")
        # self.analyze_bit_flip_effects()
        
        print("\n6. Basis comparison and key sifting")
        final_key = self.sift_key()
        
        print(f"\nKey efficiency: {len(final_key)}/{self.key_length} = {len(final_key)/self.key_length:.1%}")
        
        return final_key
    
    def demonstrate_bit_flip_comparison(self):
        """Compare protocols with and without bit flip errors"""
        print("\n" + "=" * 60)
        print("COMPARISON: WITH vs WITHOUT BIT FLIP ERRORS")
        print("=" * 60)
        
        # Store original probability
        original_prob = self.bit_flip_probability
        
        # Run without bit flip errors
        print("\n--- WITHOUT BIT FLIP ERRORS ---")
        self.bit_flip_probability = 0.0
        self.run_protocol()
        
        # Run with bit flip errors
        print("\n--- WITH BIT FLIP ERRORS ---")
        self.bit_flip_probability = original_prob
        self.run_protocol()


class BB84ComparisonProtocol:
    """Compare both bit flip and phase flip errors"""
    
    def __init__(self, key_length=8, error_probability=0.1):
        self.key_length = key_length
        self.error_probability = error_probability
        
    def run_error_comparison(self):
        """Compare bit flip vs phase flip errors"""
        print("=" * 70)
        print("COMPREHENSIVE ERROR COMPARISON: BIT FLIP vs PHASE FLIP")
        print("=" * 70)
        
        # Test Bit Flip Errors
        print("\n" + "+" * 35)
        print("TESTING BIT FLIP ERRORS")
        print("+" * 35)
        bb84_bit_flip = BB84Protocol(self.key_length, self.error_probability)
        bb84_bit_flip.run_protocol()
        
        # Test Phase Flip Errors (using the original code structure)
        print("\n" + "+" * 35)
        print("TESTING PHASE FLIP ERRORS")
        print("+" * 35)
        bb84_phase_flip = BB84PhaseFlipProtocol(self.key_length, self.error_probability)
        bb84_phase_flip.run_protocol()
        
        print("\n" + "=" * 70)
        print("ERROR TYPE SUMMARY")
        print("=" * 70)
        print("BIT FLIP ERRORS (X gate):")
        print("• Z basis (computational): |0⟩ → |1⟩, |1⟩ → |0⟩ (measurement result flips)")
        print("• X basis (superposition): |+⟩ → |+⟩, |-⟩ → |-⟩ (no measurement change)")
        print("\nPHASE FLIP ERRORS (Z gate):")
        print("• Z basis (computational): |0⟩ → |0⟩, |1⟩ → -|1⟩ (no measurement change)")
        print("• X basis (superposition): |+⟩ → |-⟩, |-⟩ → |+⟩ (measurement result flips)")
        print("\nIn BB84:")
        print("• Bit flips cause errors in Z basis measurements only")
        print("• Phase flips cause errors in X basis measurements only")
        print("• Both create detectable error patterns different from eavesdropping")


class BB84PhaseFlipProtocol(BB84Protocol):
    """Phase flip version for comparison"""
    
    def __init__(self, key_length=8, phase_flip_probability=0.0):
        super().__init__(key_length, phase_flip_probability)
        self.phase_flip_probability = phase_flip_probability
        self.phase_flip_locations = []
        
    def apply_bit_flip_error(self):
        """Override to apply phase flip instead"""
        self.phase_flip_locations = []
        
        for i in range(self.key_length):
            if random.random() < self.phase_flip_probability:
                # Apply phase flip (Z gate) to the qubit
                self.circuits[i].z(0)
                self.phase_flip_locations.append(i)
    
    def sift_key(self):
        """Modified sift key for phase flip analysis"""
        self.shared_key = []
        matching_indices = []
        
        for i in range(self.key_length):
            if self.alice_bases[i] == self.bob_bases[i]:
                self.shared_key.append(self.alice_bits[i])
                matching_indices.append(i)
        
        bob_sifted = [self.bob_results[i] for i in matching_indices]
        errors = sum(1 for a, b in zip(self.shared_key, bob_sifted) if a != b)
        error_rate = errors / len(self.shared_key) if self.shared_key else 0
        
        # Analyze which errors are due to phase flips
        phase_flip_errors = 0
        for i, idx in enumerate(matching_indices):
            if idx in self.phase_flip_locations:
                if self.alice_bases[idx] == 1:  # X basis measurement
                    phase_flip_errors += 1
                    # print(f"  Phase flip error detected at position {idx} (X basis)")
        
        print(f"\nBit error rate: {error_rate:.2%}")
        print(f"Errors due to phase flips in X basis: {phase_flip_errors}")
        print(f"Other errors: {errors - phase_flip_errors}")
        
        return self.shared_key
    
    def run_protocol(self):
        """Execute BB84 with phase flip errors"""
        print("=" * 50)
        print("BB84 QUANTUM KEY DISTRIBUTION WITH PHASE FLIP ERRORS")
        print("=" * 50)
        
        print(f"Phase flip probability: {self.phase_flip_probability:.2%}")
        
        print("\n1. Alice generates random bits and bases")
        self.generate_random_bits()
        
        print("\n2. Alice prepares and sends qubits")
        self.alice_prepare_qubits()
        
        print("\n3. Apply phase flip errors during transmission")
        self.apply_bit_flip_error()  # This now applies phase flip
        
        print("\n4. Bob measures qubits with random bases")
        self.bob_measure_qubits()
        
        print("\n5. Basis comparison and key sifting")
        final_key = self.sift_key()
        
        print(f"\nKey efficiency: {len(final_key)}/{self.key_length} = {len(final_key)/self.key_length:.1%}")
        
        return final_key


def main():
    # Demonstrate BB84 with different bit flip probabilities
    print("Testing BB84 protocol with bit flip errors\n")
    
    # Test with 20% bit flip probability
    print("Single protocol test:")
    # bb84_with_errors = BB84Protocol(key_length=20, bit_flip_probability=0.2)
    # bb84_with_errors.run_protocol()
    
    # Demonstrate comparison between bit flip and no errors
    print("\n" + "="*60)
    bb84_comparison = BB84Protocol(key_length=10000, bit_flip_probability=0.2)
    bb84_comparison.demonstrate_bit_flip_comparison()
    print(a := np.array([0,1,2]))
    # Comprehensive comparison between bit flip and phase flip
    print("\n" + "="*60)
    comparison = BB84ComparisonProtocol(key_length=10000, error_probability=0.2)
    comparison.run_error_comparison()

if __name__ == "__main__":
    main()