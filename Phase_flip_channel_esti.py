import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import random

class BB84ChannelEstimator:
    def __init__(self, num_test_bits=1000, error_prob=0.1, error_var=0.1):
        """
        Initialize BB84 channel error estimator with random error types
        
        Args:
            num_test_bits (int): Number of test bits to send
            error_prob (float): Actual error probability of the channel
            error_var (float): Variance for sampling the actual error probability
        """
        self.num_test_bits = num_test_bits
        # Error prob sampled from gaussian(error_prob, error_var)
        self.error_mean = error_prob
        self.error_var = error_var
        self.error_prob = np.random.normal(loc=error_prob, scale=error_var)
        self.simulator = AerSimulator()
        
        # Protocol data
        self.sender_bits = []
        self.sender_bases = []
        self.receiver_bases = []
        self.receiver_results = []
        self.matching_indices = []
        
        # Error tracking
        self.bit_flips_applied = 0
        self.phase_flips_applied = 0
        
    def generate_random_protocol_data(self):
        """Generate random bits and bases for sender and receiver (like BB84)"""
        # Sender (Alice) generates random bits and bases
        self.sender_bits = [random.randint(0, 1) for _ in range(self.num_test_bits)]
        self.sender_bases = [random.randint(0, 1) for _ in range(self.num_test_bits)]  # 0=Z, 1=X
        
        # Receiver (Bob) generates random measurement bases
        self.receiver_bases = [random.randint(0, 1) for _ in range(self.num_test_bits)]  # 0=Z, 1=X
        
        # Find matching basis indices
        self.matching_indices = [i for i in range(self.num_test_bits) 
                               if self.sender_bases[i] == self.receiver_bases[i]]
        
    def prepare_and_send_qubits(self):
        """Sender prepares qubits and sends through noisy channel with random error types"""
        self.circuits = []
        self.bit_flips_applied = 0
        self.phase_flips_applied = 0
        
        for i in range(self.num_test_bits):
            # Prepare qubit based on sender's bit and basis
            qc = QuantumCircuit(1, 1)
            
            # Encode bit value
            if self.sender_bits[i] == 1:
                qc.x(0)  # Apply X gate if bit is 1
                
            # Apply basis encoding
            if self.sender_bases[i] == 1:
                qc.h(0)  # Apply Hadamard if using X basis
            
            # Apply random error during transmission
            if random.random() < self.error_prob:
                # Randomly choose between bit flip (X) or phase flip (Z)
                if random.random() < 0.5:
                    qc.x(0)  # Bit flip error
                    self.bit_flips_applied += 1
                else:
                    qc.z(0)  # Phase flip error
                    self.phase_flips_applied += 1
                
            self.circuits.append(qc)
            
        return self.bit_flips_applied, self.phase_flips_applied
        
    def receiver_measure_qubits(self):
        """Receiver measures qubits using random bases"""
        self.receiver_results = []
        
        for i in range(self.num_test_bits):
            # Copy the received (potentially noisy) circuit
            qc = self.circuits[i].copy()
            
            # Apply receiver's measurement basis
            if self.receiver_bases[i] == 1:
                qc.h(0)  # Apply Hadamard if measuring in X basis
                
            # Measure
            qc.measure(0, 0)
            
            # Execute measurement
            job = self.simulator.run(qc, shots=1)
            result = job.result()
            counts = result.get_counts()
            
            # Get measurement result
            measured_bit = int(list(counts.keys())[0])
            self.receiver_results.append(measured_bit)
        
    def calculate_qber_matching_bases_only(self):
        """Calculate QBER only for bits with matching bases"""
        if not self.matching_indices:
            print("No matching bases found!")
            return 0.0
        
        matching_bits = len(self.matching_indices)
        error_count = 0
        
        # Count errors only for matching bases
        for i in self.matching_indices:
            if self.sender_bits[i] != self.receiver_results[i]:
                error_count += 1
        qber = error_count / matching_bits if matching_bits > 0 else 0
        
        return qber
        
    def run_bb84_channel_estimation(self):
        """Run complete BB84-style channel estimation"""
        # Sample new error probability for this run
        self.error_prob = np.random.normal(loc=self.error_mean, scale=self.error_var)
        
        # Ensure error probability is non-negative
        self.error_prob = max(0, self.error_prob)

        # Generate random bits and bases
        self.generate_random_protocol_data()
        
        # Prepare qubits and send through noisy channel
        bit_flips, phase_flips = self.prepare_and_send_qubits()
        
        # Receiver measures with random bases
        self.receiver_measure_qubits()
        
        # Calculate QBER for matching bases
        qber = self.calculate_qber_matching_bases_only()
        
        return qber
        
    def detailed_basis_analysis(self):
        """Provide detailed analysis of errors by basis type"""
        print(f"\nDetailed Basis Analysis:")
        
        z_basis_matches = [i for i in self.matching_indices if self.sender_bases[i] == 0]
        x_basis_matches = [i for i in self.matching_indices if self.sender_bases[i] == 1]
        
        # Z-basis errors
        z_errors = sum(1 for i in z_basis_matches 
                      if self.sender_bits[i] != self.receiver_results[i])
        z_qber = z_errors / len(z_basis_matches) if z_basis_matches else 0
        
        # X-basis errors  
        x_errors = sum(1 for i in x_basis_matches 
                      if self.sender_bits[i] != self.receiver_results[i])
        x_qber = x_errors / len(x_basis_matches) if x_basis_matches else 0
        
        print(f"Z-basis (computational): {len(z_basis_matches)} bits, {z_errors} errors, QBER = {z_qber:.4f}")
        print(f"X-basis (Hadamard): {len(x_basis_matches)} bits, {x_errors} errors, QBER = {x_qber:.4f}")
        print(f"Error breakdown: {self.bit_flips_applied} bit flips, {self.phase_flips_applied} phase flips")
        
        return z_qber, x_qber

    def get_error_statistics(self):
        """Return statistics about the errors applied"""
        total_errors = self.bit_flips_applied + self.phase_flips_applied
        return {
            'total_errors': total_errors,
            'bit_flips': self.bit_flips_applied,
            'phase_flips': self.phase_flips_applied,
            'bit_flip_ratio': self.bit_flips_applied / total_errors if total_errors > 0 else 0,
            'phase_flip_ratio': self.phase_flips_applied / total_errors if total_errors > 0 else 0
        }

def run_qber_analysis_multiple_rates():
    """Analyze QBER estimation accuracy for different error rates"""
    print("\n" + "=" * 60)
    print("QBER ESTIMATION ACCURACY ANALYSIS")
    print("=" * 60)
    
    true_rates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    estimated_qbers = []
    
    for true_rate in true_rates:
        print(f"\n--- Testing true error rate: {true_rate} ---")
        estimator = BB84ChannelEstimator(num_test_bits=2000, error_prob=true_rate)
        qber = estimator.run_bb84_channel_estimation()
        estimated_qbers.append(qber)
        
        # Show error statistics
        error_stats = estimator.get_error_statistics()
        print(f"Errors applied: {error_stats['total_errors']} "
              f"({error_stats['bit_flips']} bit flips, {error_stats['phase_flips']} phase flips)")
        
        # Also show basis-specific analysis
        estimator.detailed_basis_analysis()
    
    print("\n" + "=" * 50)
    print("ESTIMATION ACCURACY SUMMARY")
    print("=" * 50)
    print("True Rate | Estimated QBER | Difference")
    print("-" * 40)
    for true, est in zip(true_rates, estimated_qbers):
        diff = abs(true - est)
        print(f"{true:8.3f} | {est:14.4f} | {diff:10.4f}")

# Example usage
def main():
    # Single channel estimation
    estimator = BB84ChannelEstimator(num_test_bits=10000, error_prob=0.2, error_var=0.02)
    qber = estimator.run_bb84_channel_estimation()
    
    # Show error statistics
    error_stats = estimator.get_error_statistics()
    print(f"\nError Statistics:")
    print(f"Total errors applied: {error_stats['total_errors']}")
    print(f"Bit flips: {error_stats['bit_flips']} ({error_stats['bit_flip_ratio']:.1%})")
    print(f"Phase flips: {error_stats['phase_flips']} ({error_stats['phase_flip_ratio']:.1%})")
    print(f"Estimated QBER: {qber:.4f}")
    
    estimator.detailed_basis_analysis()
    
    # Run analysis for multiple error rates
    # run_qber_analysis_multiple_rates()  # Uncomment for full analysis

if __name__ == "__main__":
    main()