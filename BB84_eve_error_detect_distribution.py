import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import random
import matplotlib.pyplot as plt
from Phase_flip_channel_esti import BB84ChannelEstimator

class BB84Protocol:
    def __init__(self, num_bits=1000, channel_noise_mean=0.05, channel_noise_std=0.1, eve_intercept_prob=0.0, eve_strategy='random',
                 estimate_mean_qber=0.0, estimate_std_qber=0.0):
        """
        Complete BB84 Quantum Key Distribution Protocol

        Args:
            num_bits (int): Number of bits for key generation
            channel_noise_prob (float): Probability of channel noise (phase flips)
            eve_intercept_prob (float): Probability Eve intercepts each qubit
            eve_strategy (str): Eve's strategy - 'random', 'z_basis', 'x_basis', 'optimal'
        """
        self.num_bits = num_bits
        self.channel_noise_mean = channel_noise_mean
        self.channel_noise_std = channel_noise_std
        # Convert mean and std to probability
        self.channel_noise_prob = np.random.normal(loc=channel_noise_mean, scale=channel_noise_std)
        self.eve_intercept_prob = eve_intercept_prob
        self.eve_strategy = eve_strategy
        self.simulator = AerSimulator()
        self.estimate_mean_qber = estimate_mean_qber
        self.estimate_std_qber = estimate_std_qber
        # Alice's data
        self.alice_bits = []
        self.alice_bases = []
        
        # Bob's data
        self.bob_bases = []
        self.bob_results = []
        
        # Eve's data
        self.eve_bases = []
        self.eve_results = []
        self.eve_intercepts = []  # Which qubits Eve intercepted
        
        # Protocol results
        self.matching_indices = []
        self.sifted_key_alice = []
        self.sifted_key_bob = []
        self.final_key_alice = []
        self.final_key_bob = []
        
        # Statistics
        self.qber = 0.0
        self.eve_detected = False
        self.key_generation_successful = False
        
    def step1_alice_prepares_qubits(self):
        """Step 1: Alice generates random bits and bases, prepares qubits"""
        print("=== STEP 1: Alice Prepares Qubits ===")
        
        # Alice generates random bits and bases
        self.alice_bits = [random.randint(0, 1) for _ in range(self.num_bits)]
        self.alice_bases = [random.randint(0, 1) for _ in range(self.num_bits)]  # 0=Z, 1=X
        
        print(f"Alice generated {self.num_bits} random bits")
        print(f"Alice chose random bases (Z=0, X=1)")
        print(f"Sample: bits={self.alice_bits[:10]}, bases={self.alice_bases[:10]}")
        
        # Prepare quantum circuits
        self.circuits = []
        for i in range(self.num_bits):
            qc = QuantumCircuit(1, 1)
            
            # Encode Alice's bit
            if self.alice_bits[i] == 1:
                qc.x(0)  # |1⟩ state
            # else: qubit starts in |0⟩
            
            # Apply Alice's basis choice
            if self.alice_bases[i] == 1:  # X basis
                qc.h(0)  # Hadamard for X basis encoding
            # Z basis needs no additional gate
            
            self.circuits.append(qc)
            
        print("Alice prepared quantum states and sent them to Bob")
        
    def step2_eve_eavesdrops(self):
        """Step 2: Eve potentially intercepts and measures qubits"""
        print(f"\n=== STEP 2: Eve's Eavesdropping (Strategy: {self.eve_strategy}) ===")
        
        self.eve_intercepts = []
        self.eve_bases = []
        self.eve_results = []
        eve_interceptions = 0
        
        for i in range(self.num_bits):
            # Decide if Eve intercepts this qubit
            if random.random() < self.eve_intercept_prob:
                self.eve_intercepts.append(True)
                eve_interceptions += 1
                
                # Choose Eve's measurement basis based on strategy
                if self.eve_strategy == 'random':
                    eve_basis = random.randint(0, 1)
                elif self.eve_strategy == 'z_basis':
                    eve_basis = 0  # Always Z basis
                elif self.eve_strategy == 'x_basis':
                    eve_basis = 1  # Always X basis
                elif self.eve_strategy == 'optimal':
                    # In real scenario, Eve doesn't know Alice's basis
                    # This is just for comparison
                    eve_basis = self.alice_bases[i]
                else:
                    eve_basis = random.randint(0, 1)
                    
                self.eve_bases.append(eve_basis)
                
                # Eve measures the qubit
                qc_eve = self.circuits[i].copy()
                
                # Apply Eve's measurement basis
                if eve_basis == 1:  # X basis measurement
                    qc_eve.h(0)
                    
                qc_eve.measure(0, 0)
                
                # Execute Eve's measurement
                job = self.simulator.run(qc_eve, shots=1)
                result = job.result()
                counts = result.get_counts()
                eve_result = int(list(counts.keys())[0])
                self.eve_results.append(eve_result)
                
                # Eve prepares a new qubit to send to Bob based on her measurement
                qc_new = QuantumCircuit(1, 1)
                if eve_result == 1:
                    qc_new.x(0)
                if eve_basis == 1:  # If Eve measured in X basis, prepare in X basis
                    qc_new.h(0)
                # Replace the circuit Bob will receive
                self.circuits[i] = qc_new
                
            else:
                self.eve_intercepts.append(False)
                self.eve_bases.append(None)
                self.eve_results.append(None)
        
        print(f"Eve intercepted {eve_interceptions}/{self.num_bits} qubits ({eve_interceptions/self.num_bits*100:.1f}%)")
        if eve_interceptions > 0:
            print(f"Eve's measurement bases: {[b for b in self.eve_bases if b is not None][:10]}")
            
    def step3_apply_channel_noise(self):
        """Step 3: Apply channel noise (phase flips)"""
        print(f"\n=== STEP 3: Channel Noise (Phase flip probability: {self.channel_noise_prob}) ===")
        print(f"Applying phase flips with probability {self.channel_noise_prob:.2f}")
        noise_applications = 0
        for i in range(self.num_bits):
            if random.random() < self.channel_noise_prob:
                self.circuits[i].z(0)  # Apply phase flip
                noise_applications += 1
                
        print(f"Channel applied {noise_applications}/{self.num_bits} phase flip errors")
        
    def step4_bob_measures(self):
        """Step 4: Bob chooses random bases and measures"""
        print(f"\n=== STEP 4: Bob Measures Qubits ===")
        
        # Bob chooses random measurement bases
        self.bob_bases = [random.randint(0, 1) for _ in range(self.num_bits)]
        self.bob_results = []
        
        for i in range(self.num_bits):
            qc = self.circuits[i].copy()
            
            # Apply Bob's measurement basis
            if self.bob_bases[i] == 1:  # X basis measurement
                qc.h(0)
                
            qc.measure(0, 0)
            
            # Execute measurement
            job = self.simulator.run(qc, shots=1)
            result = job.result()
            counts = result.get_counts()
            bob_result = int(list(counts.keys())[0])
            self.bob_results.append(bob_result)
            
        print(f"Bob measured all {self.num_bits} qubits with random bases")
        print(f"Sample Bob bases: {self.bob_bases[:10]}")
        print(f"Sample Bob results: {self.bob_results[:10]}")
        
    def step5_basis_reconciliation(self):
        """Step 5: Alice and Bob compare bases publicly"""
        print(f"\n=== STEP 5: Basis Reconciliation ===")
        
        # Find matching bases
        self.matching_indices = []
        for i in range(self.num_bits):
            if self.alice_bases[i] == self.bob_bases[i]:
                self.matching_indices.append(i)
                
        print(f"Matching bases: {len(self.matching_indices)}/{self.num_bits} ({len(self.matching_indices)/self.num_bits*100:.1f}%)")
        
        # Create sifted key from matching bases
        self.sifted_key_alice = [self.alice_bits[i] for i in self.matching_indices]
        self.sifted_key_bob = [self.bob_results[i] for i in self.matching_indices]
        
        print(f"Sifted key length: {len(self.sifted_key_alice)} bits")
        
    def step6_error_detection(self, test_fraction=0.5):
        """Step 6: Randomly test a fraction of sifted key for errors"""
        print(f"\n=== STEP 6: Error Detection (Testing {test_fraction*100:.1f}% of sifted key) ===")
        
        if len(self.sifted_key_alice) == 0:
            print("No sifted key available for testing!")
            return False
            
        # Randomly select test bits
        num_test_bits = max(1, int(len(self.sifted_key_alice) * test_fraction))
        test_indices = random.sample(range(len(self.sifted_key_alice)), num_test_bits)
        
        # Count errors in test bits
        errors = 0
        for i in test_indices:
            if self.sifted_key_alice[i] != self.sifted_key_bob[i]:
                errors += 1
                
        self.qber = errors / num_test_bits if num_test_bits > 0 else 0
        
        print(f"Tested {num_test_bits} bits, found {errors} errors")
        print(f"QBER: {self.qber:.4f} ({self.qber*100:.2f}%)")
        
        # Security threshold (typically 11% for BB84)
        ##if (2mu-1)(2x-1)<0 then no eve
        multip = (self.estimate_mean_qber * 2 - 1)*(2*self.qber - 1)
        if multip > 0:
            print("✅ No Eve detected - keys are secure")
            self.eve_detected = False
            return True
        else:
            print("❌ Eve detected - keys may be compromised")
            self.eve_detected = True
            return False
    def step7_privacy_amplification(self, test_fraction=0.1):
        """Step 7: Remove test bits and create final key"""
        print(f"\n=== STEP 7: Privacy Amplification ===")
        
        if self.eve_detected:
            print("Eve detected - no final key generated")
            return
            
        # Remove test bits (in practice, use more sophisticated error correction)
        num_test_bits = max(1, int(len(self.sifted_key_alice) * test_fraction))
        remaining_bits = len(self.sifted_key_alice) - num_test_bits
        
        # In real implementation, would use proper error correction and privacy amplification
        # Here we just take the remaining bits after removing test bits
        self.final_key_alice = self.sifted_key_alice[:remaining_bits]
        self.final_key_bob = self.sifted_key_bob[:remaining_bits]
        
        # Check if final keys match (they should if no errors)
        key_matches = (self.final_key_alice == self.final_key_bob)
        
        print(f"Final key length: {len(self.final_key_alice)} bits")
        print(f"Keys match: {key_matches}")
        
        if key_matches:
            print("✅ Secure key successfully generated!")
            self.key_generation_successful = True
        else:
            print("❌ Key generation failed - keys don't match")
            
    def run_complete_bb84_protocol(self):
        """Run the complete BB84 protocol"""
        print("🔒 " + "="*60)
        print("    BB84 QUANTUM KEY DISTRIBUTION PROTOCOL")
        print("="*63)
        print(f"Parameters:")
        print(f"  • Number of qubits: {self.num_bits}")
        print(f"  • Channel noise probability: {self.channel_noise_prob}")
        print(f"  • Eve intercept probability: {self.eve_intercept_prob}")
        print(f"  • Eve's strategy: {self.eve_strategy}")
        
        # Execute all steps
        self.step1_alice_prepares_qubits()
        self.step2_eve_eavesdrops()
        self.step3_apply_channel_noise()
        self.step4_bob_measures()
        self.step5_basis_reconciliation()
        
        # Error detection and key generation
        if self.step6_error_detection():
            self.step7_privacy_amplification()
        
        self.print_final_summary()
        
    def print_final_summary(self):
        """Print final protocol summary"""
        print(f"\n🔒 " + "="*50)
        print("PROTOCOL SUMMARY")
        print("="*53)
        
        print(f"Original qubits sent: {self.num_bits}")
        print(f"Sifted key length: {len(self.sifted_key_alice)}")
        print(f"Final key length: {len(self.final_key_alice) if self.final_key_alice else 0}")
        print(f"QBER: {self.qber:.4f} ({self.qber*100:.2f}%)")
        print(f"Eve detected: {'Yes' if self.eve_detected else 'No'}")
        print(f"Key generation successful: {'Yes' if self.key_generation_successful else 'No'}")
        
        if self.eve_intercept_prob > 0:
            eve_interceptions = sum(self.eve_intercepts)
            print(f"Eve intercepted: {eve_interceptions}/{self.num_bits} qubits")
            
    def analyze_eve_detection_probability(self):
        """Analyze how Eve's presence affects QBER"""
        print(f"\n=== EVE DETECTION ANALYSIS ===")
        
        if self.eve_intercept_prob == 0:
            print("No eavesdropping to analyze")
            return
            
        # Calculate expected QBER due to Eve
        # When Eve measures in wrong basis, she introduces 25% error rate
        expected_eve_error_rate = self.eve_intercept_prob * 0.25
        
        print(f"Expected QBER from Eve alone: {expected_eve_error_rate:.4f}")
        print(f"Expected QBER from channel noise: {self.channel_noise_prob:.4f}")
        print(f"Expected total QBER: {expected_eve_error_rate + self.channel_noise_prob:.4f}")
        print(f"Measured QBER: {self.qber:.4f}")


def run_bb84_scenarios():
    """Run BB84 under different scenarios"""
    print("🔬 TESTING DIFFERENT BB84 SCENARIOS\n")
    scenarios = [
        {"name": "No Eve, No Noise", "noise_mean": 0.2, "noise_std": 0.1, "eve_prob": 1.0, "eve_strategy": "random"},
        # {"name": "Channel Noise Only", "noise": 0.05, "eve_prob": 0.0, "eve_strategy": "random"},
        # {"name": "Passive Eve (Random)", "noise": 0.02, "eve_prob": 0.3, "eve_strategy": "random"},
        # {"name": "Passive Eve (Z-basis)", "noise": 0.02, "eve_prob": 0.3, "eve_strategy": "z_basis"},
        # {"name": "Strong Eavesdropping", "noise": 0.02, "eve_prob": 0.8, "eve_strategy": "random"},
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario['name']}")
        print('='*60)
        estimator = BB84ChannelEstimator(num_test_bits=1000, error_prob=scenario["noise_mean"], error_var=scenario["noise_std"])
        # Initialize an empty array for QBER values
        qber_array = np.array([])
        for i in range(1000):
            qber = estimator.run_bb84_channel_estimation()
            qber_array = np.append(qber_array, qber)
        print(qber_array)
        # Calculate the mean and standard deviation of QBER
        mean_qber = np.mean(qber_array)*2
        std_qber = np.std(qber_array)*2
        print(f"Mean QBER: {mean_qber:.4f}, Std Dev: {std_qber:.4f}")
        
        bb84 = BB84Protocol(
            num_bits=128,
            channel_noise_mean=scenario['noise_mean'],
            channel_noise_std=scenario['noise_std'],
            eve_intercept_prob=scenario['eve_prob'],
            eve_strategy=scenario['eve_strategy'],
            estimate_mean_qber=mean_qber,
            estimate_std_qber=std_qber
        )
        
        bb84.run_complete_bb84_protocol()
        bb84.analyze_eve_detection_probability()


def main():
    """Main function to demonstrate BB84"""
    
    # Single protocol run
    print("Single BB84 Protocol Run:")
    bb84 = BB84Protocol(
        num_bits=1000,
        channel_noise_prob=0.03,
        eve_intercept_prob=0.2,
        eve_strategy='random'
    )
    bb84.run_complete_bb84_protocol()
    bb84.analyze_eve_detection_probability()
    
    # Multiple scenarios
    print("\n" * 3)
    run_bb84_scenarios()


if __name__ == "__main__":
    run_bb84_scenarios()
    # main()