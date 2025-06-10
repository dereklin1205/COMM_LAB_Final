import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import random
import matplotlib.pyplot as plt
from scipy import stats
import math

class RandomBasisQBERTest:
    def __init__(self, num_bits=1000, channel_noise_mean=0.05, channel_noise_std=0.1, eve_detect_prob=0.7):
        """
        Random basis QKD simulation with Eve eavesdropping
        
        Args:
            num_bits: Number of qubits to test
            channel_noise_mean: Mean of channel noise (μ)
            channel_noise_std: Standard deviation of channel noise (σ)
            eve_detect_prob: Probability that Eve attempts to eavesdrop (0.0 to 1.0)
        """
        self.num_bits = num_bits
        self.channel_noise_mean = channel_noise_mean
        self.channel_noise_std = channel_noise_std
        self.eve_detect_prob = eve_detect_prob
        self.simulator = AerSimulator()
        
        # Calculate theoretical P1 (error probability without Eve)
        self.theoretical_p1_no_eve = self._calculate_theoretical_p1()
        # Theoretical P1 with Eve (random basis eavesdropping)
        self.theoretical_p1_with_eve = self._calculate_theoretical_p1_with_eve()
        
        print(f"Theoretical P1 (no Eve): {self.theoretical_p1_no_eve:.4f}")
        print(f"Theoretical P1 (with Eve): {self.theoretical_p1_with_eve:.4f}")
        print(f"Eve detection probability: {self.eve_detect_prob:.1f}")
    
    def _calculate_theoretical_p1(self):
        """Calculate theoretical P1 = P(Y <= X) where X~N(μ,σ²), Y~U(0,1)"""
        mu = self.channel_noise_mean
        sigma = self.channel_noise_std
        
        from scipy import integrate
        
        def integrand1(x):
            return x * stats.norm.pdf(x, mu, sigma)
        
        # First integral: ∫₀¹ x * φ((x-μ)/σ)/σ dx
        integral1, _ = integrate.quad(integrand1, 0, 1)
        
        # Second integral: ∫₁^∞ φ((x-μ)/σ)/σ dx = 1 - Φ((1-μ)/σ)
        integral2 = 1 - stats.norm.cdf(1, mu, sigma)
        
        p1 = integral1 + integral2
        return p1/2
    
    def _calculate_theoretical_p1_with_eve(self):
        """
        Calculate theoretical P1 with Eve eavesdropping
        With random basis eavesdropping and detection probability,
        Eve introduces additional error only when she's active.
        """
        # Base error from channel noise
        p_channel = self.theoretical_p1_no_eve
        
        # Additional error from Eve's random basis eavesdropping when she's active
        # When Alice and Bob use same basis and Eve uses wrong basis,
        # she introduces 50% error in those qubits
        # Eve chooses wrong basis 50% of the time, but is only active eve_detect_prob of the time
        p_eve_when_active = 0.25  # 50% wrong basis * 50% error when wrong
        p_eve = self.eve_detect_prob * p_eve_when_active
        
        # Combined error probability (assuming independent errors)
        # P(error) = P(channel error) + P(eve error) - P(both errors)
        p_combined = p_channel + p_eve - 2*(p_channel * p_eve)
        
        return p_combined
    
    def run_random_basis_test(self, num_runs=1000, include_eve=True):
        """
        Run random basis QKD test with optional Eve eavesdropping
        
        Args:
            num_runs: Number of test runs
            include_eve: Whether to include Eve eavesdropping capability
        """
        print(f"\n{'='*60}")
        print(f"RANDOM BASIS QKD SIMULATION")
        print(f"{'='*60}")
        print(f"Channel noise: N({self.channel_noise_mean}, {self.channel_noise_std}²)")
        print(f"Number of runs: {num_runs}")
        print(f"Bits per run: {self.num_bits}")
        print(f"Eve eavesdropping enabled: {'Yes' if include_eve else 'No'}")
        if include_eve:
            print(f"Eve detection probability: {self.eve_detect_prob:.1f}")
        
        qber_results = []
        sifted_key_lengths = []
        eve_active_count = 0
        
        for run in range(num_runs):
            qber, sifted_length, eve_was_active = self._simulate_qkd_protocol(include_eve)
            qber_results.append(qber)
            sifted_key_lengths.append(sifted_length)
            if eve_was_active:
                eve_active_count += 1
            
            if (run + 1) % 100 == 0:
                print(f"Completed {run + 1}/{num_runs} runs...")
        
        # Statistical analysis
        mean_qber = np.mean(qber_results)
        std_qber = np.std(qber_results)
        mean_sifted_length = np.mean(sifted_key_lengths)
        
        theoretical = self.theoretical_p1_with_eve if include_eve else self.theoretical_p1_no_eve
        
        results = {
            'qber_values': qber_results,
            'sifted_lengths': sifted_key_lengths,
            'mean_qber': mean_qber,
            'std_qber': std_qber,
            'mean_sifted_length': mean_sifted_length,
            'theoretical_qber': theoretical,
            'include_eve': include_eve,
            'eve_active_rate': eve_active_count / num_runs if include_eve else 0
        }
        
        print(f"\nResults:")
        print(f"Observed QBER: {mean_qber:.4f} ± {std_qber:.4f}")
        print(f"Theoretical QBER: {theoretical:.4f}")
        print(f"Difference: {abs(mean_qber - theoretical):.4f}")
        print(f"Average sifted key length: {mean_sifted_length:.1f} bits ({mean_sifted_length/self.num_bits*100:.1f}%)")
        if include_eve:
            print(f"Eve was active in {eve_active_count}/{num_runs} runs ({eve_active_count/num_runs*100:.1f}%)")
        
        # Statistical significance test
        self._perform_statistical_analysis(results)
        
        return results
    
    def _simulate_qkd_protocol(self, include_eve=True):
        """Simulate one complete QKD protocol run"""
        
        # Determine if Eve is active in this run
        eve_is_active = include_eve and (random.random() < self.eve_detect_prob)
        
        # Step 1: Alice prepares qubits with random bits and bases
        alice_bits = [random.randint(0, 1) for _ in range(self.num_bits)]
        alice_bases = [random.randint(0, 1) for _ in range(self.num_bits)]  # 0=Z, 1=X
        
        circuits = []
        for i in range(self.num_bits):
            qc = QuantumCircuit(1, 1)
            
            # Encode Alice's bit
            if alice_bits[i] == 1:
                qc.x(0)
            
            # Apply Alice's basis
            if alice_bases[i] == 1:  # X basis
                qc.h(0)
            
            circuits.append(qc)
        
        # Step 2: Eve's eavesdropping (if active)
        eve_measurements = []
        if eve_is_active:
            eve_bases = [random.randint(0, 1) for _ in range(self.num_bits)]
            
            for i in range(self.num_bits):
                qc = circuits[i].copy()
                
                # Eve measures in her chosen basis
                if eve_bases[i] == 1:  # X basis
                    qc.h(0)
                
                qc.measure(0, 0)
                
                # Execute Eve's measurement
                job = self.simulator.run(qc, shots=1)
                result = job.result()
                counts = result.get_counts()
                eve_result = int(list(counts.keys())[0])
                eve_measurements.append(eve_result)
                
                # Eve prepares a new qubit based on her measurement
                new_qc = QuantumCircuit(1, 1)
                if eve_result == 1:
                    new_qc.x(0)
                
                # Apply Eve's basis (same as her measurement basis)
                if eve_bases[i] == 1:  # X basis
                    new_qc.h(0)
                
                circuits[i] = new_qc
        
        # Step 3: Apply channel noise (both bit flip and phase flip)
        channel_noise_prob = np.random.normal(self.channel_noise_mean, self.channel_noise_std)
        channel_noise_prob = max(0, min(1, channel_noise_prob))  # Clamp to [0,1]
        
        for i in range(self.num_bits):
            if random.random() < channel_noise_prob:
                # Randomly choose between bit flip and phase flip
                noise_type = random.choice(['bit_flip', 'phase_flip'])
                
                if noise_type == 'bit_flip':
                    circuits[i].x(0)  # Pauli-X (bit flip)
                else:
                    circuits[i].z(0)  # Pauli-Z (phase flip)
        
        # Step 4: Bob measures with random bases
        bob_bases = [random.randint(0, 1) for _ in range(self.num_bits)]
        bob_results = []
        
        for i in range(self.num_bits):
            qc = circuits[i].copy()
            
            # Apply Bob's measurement basis
            if bob_bases[i] == 1:  # X basis
                qc.h(0)
            
            qc.measure(0, 0)
            
            # Execute Bob's measurement
            job = self.simulator.run(qc, shots=1)
            result = job.result()
            counts = result.get_counts()
            bob_result = int(list(counts.keys())[0])
            bob_results.append(bob_result)
        
        # Step 5: Basis sifting (keep only bits where Alice and Bob used same basis)
        sifted_alice_bits = []
        sifted_bob_bits = []
        
        for i in range(self.num_bits):
            if alice_bases[i] == bob_bases[i]:  # Same basis
                sifted_alice_bits.append(alice_bits[i])
                sifted_bob_bits.append(bob_results[i])
        
        # Step 6: Calculate QBER on sifted key
        if len(sifted_alice_bits) == 0:
            return 0.0, 0, eve_is_active  # No sifted bits
        
        errors = sum(1 for i in range(len(sifted_alice_bits)) 
                    if sifted_alice_bits[i] != sifted_bob_bits[i])
        qber = errors / len(sifted_alice_bits)
        
        return qber, len(sifted_alice_bits), eve_is_active
    
    def _perform_statistical_analysis(self, results):
        """Perform statistical tests to compare observed vs theoretical"""
        
        print(f"\n{'='*40}")
        print("STATISTICAL ANALYSIS")
        print(f"{'='*40}")
        
        observed_mean = results['mean_qber']
        theoretical = results['theoretical_qber']
        observed_std = results['std_qber']
        n = len(results['qber_values'])
        
        # One-sample t-test against theoretical value
        t_stat, p_value = stats.ttest_1samp(results['qber_values'], theoretical)
        
        # 95% confidence interval
        ci_margin = 1.96 * observed_std / math.sqrt(n)
        ci_lower = observed_mean - ci_margin
        ci_upper = observed_mean + ci_margin
        
        print(f"Observed QBER: {observed_mean:.6f} ± {observed_std:.6f}")
        print(f"Theoretical QBER: {theoretical:.6f}")
        print(f"Difference: {abs(observed_mean - theoretical):.6f}")
        print(f"95% CI: [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print("⚠️  Significant difference from theoretical value")
        else:
            print("✅ Matches theoretical value (p > 0.05)")
        
        # Check if theoretical value is in confidence interval
        if ci_lower <= theoretical <= ci_upper:
            print("✅ Theoretical value within 95% CI")
        else:
            print("⚠️  Theoretical value outside 95% CI")
    
    def _plot_comparison(self, results_no_eve, results_with_eve):
        """Plot comparison between scenarios with and without Eve"""
        ##one plot only
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        # fig, (ax1, ax2) = plt.subplots(1, 1, figsize=(15, 6))
        
        # QBER comparison
        ax1.hist(results_no_eve['qber_values'], bins=30, alpha=0.7, density=True, 
                label='No Eve', color='blue')
        ax1.hist(results_with_eve['qber_values'], bins=30, alpha=0.7, density=True, 
                label='With Eve', color='red')
        
        ax1.axvline(results_no_eve['theoretical_qber'], color='blue', linestyle='--', 
                   label=f'Theory (No Eve): {results_no_eve["theoretical_qber"]:.4f}')
        ax1.axvline(results_with_eve['theoretical_qber'], color='red', linestyle='--', 
                   label=f'Theory (With Eve): {results_with_eve["theoretical_qber"]:.4f}')
        ax1.axvline(results_no_eve['mean_qber'], color='blue', linestyle=':',
                   label=f'Mean (No Eve): {results_no_eve["mean_qber"]:.4f}')
        ax1.axvline(results_with_eve['mean_qber'], color='red', linestyle=':',
                   label=f'Mean (With Eve): {results_with_eve["mean_qber"]:.4f}')

        ax1.set_xlabel('QBER')
        ax1.set_ylabel('Density')
        ax1.set_title('QBER Distribution Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Sifted key length comparison
        # ax2.hist(results_no_eve['sifted_lengths'], bins=30, alpha=0.7, density=True, 
        #         label='No Eve', color='blue')
        # ax2.hist(results_with_eve['sifted_lengths'], bins=30, alpha=0.7, density=True, 
        #         label='With Eve', color='red')
        
        # ax2.set_xlabel('Sifted Key Length')
        # ax2.set_ylabel('Density')
        # ax2.set_title('Sifted Key Length Distribution')
        # ax2.legend()
        # ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_comparison_test(self, num_runs=500):
        """Run comparison test between scenarios with and without Eve"""
        
        print("Running comparison test: No Eve vs With Eve")
        
        # Test without Eve
        print("\n" + "="*50)
        print("SCENARIO 1: NO EVE EAVESDROPPING")
        print("="*50)
        results_no_eve = self.run_random_basis_test(num_runs=num_runs, include_eve=False)
        
        # Test with Eve
        print("\n" + "="*50)
        print("SCENARIO 2: WITH EVE EAVESDROPPING")
        print("="*50)
        results_with_eve = self.run_random_basis_test(num_runs=num_runs, include_eve=True)
        
        # Comparison summary
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Scenario':<25} {'QBER':<15} {'Theoretical':<15} {'Sifted Length':<15} {'Eve Active Rate':<15}")
        print("-" * 85)
        print(f"{'No Eve':<25} {results_no_eve['mean_qber']:<15.4f} "
              f"{results_no_eve['theoretical_qber']:<15.4f} "
              f"{results_no_eve['mean_sifted_length']:<15.1f} "
              f"{results_no_eve['eve_active_rate']*100:<15.1f}%")
        print(f"{'With Eve':<25} {results_with_eve['mean_qber']:<15.4f} "
              f"{results_with_eve['theoretical_qber']:<15.4f} "
              f"{results_with_eve['mean_sifted_length']:<15.1f} "
              f"{results_with_eve['eve_active_rate']*100:<15.1f}%")
        
        # Statistical test for difference
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(results_no_eve['qber_values'], 
                                   results_with_eve['qber_values'])
        
        print(f"\nStatistical test for QBER difference:")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.2e}")
        
        if p_value < 0.001:
            print("✅ Highly significant difference detected (p < 0.001)")
        else:
            print("⚠️  No significant difference detected")
        
        # Plot comparison
        self._plot_comparison(results_no_eve, results_with_eve)
        
        return results_no_eve, results_with_eve


def main():
    """Main function to run the QKD simulation"""
    
    # Test different noise configurations
    test_configs = [
        # {"noise_mean": 0.05, "noise_std": 0.02, "name": "Low Noise"},
        # {"noise_mean": 0.15, "noise_std": 0.05, "name": "Medium Noise"},
        {"noise_mean": 0.25, "noise_std": 0.02, "name": "High Noise"}
    ]
    
    for config in test_configs:
        print(f"\n{'#'*80}")
        print(f"TESTING CONFIGURATION: {config['name']}")
        print(f"Noise parameters: μ={config['noise_mean']}, σ={config['noise_std']}")
        print(f"{'#'*80}")
        
        # Create test instance
        test = RandomBasisQBERTest(
            num_bits=1000,
            channel_noise_mean=config['noise_mean'],
            channel_noise_std=config['noise_std'],
            eve_detect_prob=0.7  # Eve detection probability
        )
        
        # Run comparison test
        results_no_eve, results_with_eve = test.run_comparison_test(num_runs=300)
        
        print(f"\n{'-'*60}")


if __name__ == "__main__":
    main()