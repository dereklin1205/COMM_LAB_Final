import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import random
import matplotlib.pyplot as plt
from scipy import stats
import math

class FixedBasisQBERTest:
    def __init__(self, num_bits=1000, channel_noise_mean=0.05, channel_noise_std=0.1):
        """
        Fixed basis test to verify QBER theory (without Eve simulation)
        
        Args:
            num_bits: Number of qubits to test
            channel_noise_mean: Mean of channel noise (μ)
            channel_noise_std: Standard deviation of channel noise (σ)
        """
        self.num_bits = num_bits
        self.channel_noise_mean = channel_noise_mean
        self.channel_noise_std = channel_noise_std
        self.simulator = AerSimulator()
        
        # Calculate theoretical P1 (error probability without Eve)
        self.theoretical_p1 = self._calculate_theoretical_p1()
        print(f"Theoretical P1 (no Eve): {self.theoretical_p1:.4f}")
    
    def _calculate_theoretical_p1(self):
        """Calculate theoretical P1 = P(Y <= X) where X~N(μ,σ²), Y~U(0,1)"""
        mu = self.channel_noise_mean
        sigma = self.channel_noise_std
        
        # Numerical integration for P(Y <= X)
        # P1 = ∫₀¹ x * φ((x-μ)/σ) dx + ∫₁^∞ φ((x-μ)/σ) dx
        
        # Use numerical integration
        from scipy import integrate
        
        def integrand1(x):
            # x * normal_pdf(x, mu, sigma) for x in [0,1]
            return x * stats.norm.pdf(x, mu, sigma)
        
        # First integral: ∫₀¹ x * φ((x-μ)/σ)/σ dx
        integral1, _ = integrate.quad(integrand1, 0, 1)
        
        # Second integral: ∫₁^∞ φ((x-μ)/σ)/σ dx = 1 - Φ((1-μ)/σ)
        integral2 = 1 - stats.norm.cdf(1, mu, sigma)
        
        p1 = integral1 + integral2
        return p1
    
    def run_fixed_basis_test(self, fixed_basis=0, num_runs=1000):
        """
        Run fixed basis test (no Eve simulation)
        
        Args:
            fixed_basis: 0 for Z-basis, 1 for X-basis
            num_runs: Number of test runs
        """
        print(f"\n{'='*60}")
        print(f"FIXED BASIS QBER VERIFICATION TEST (NO EVE)")
        print(f"{'='*60}")
        print(f"Basis: {'Z-basis' if fixed_basis == 0 else 'X-basis'}")
        print(f"Channel noise: N({self.channel_noise_mean}, {self.channel_noise_std}²)")
        print(f"Number of runs: {num_runs}")
        print(f"Bits per run: {self.num_bits}")
        
        # Only test scenario without Eve
        print(f"\n--- Testing: Channel Noise Only ---")
        
        qber_results = []
        for run in range(num_runs):
            qber = self._simulate_transmission(fixed_basis)
            qber_results.append(qber)
            
            if (run + 1) % 100 == 0:
                print(f"Completed {run + 1}/{num_runs} runs...")
        
        # Statistical analysis
        mean_qber = np.mean(qber_results)
        std_qber = np.std(qber_results)
        
        results = {
            'Channel Noise Only': {
                'qber_values': qber_results,
                'mean': mean_qber,
                'std': std_qber,
                'theoretical': self.theoretical_p1
            }
        }
        
        print(f"Observed QBER: {mean_qber:.4f} ± {std_qber:.4f}")
        print(f"Theoretical QBER: {self.theoretical_p1:.4f}")
        print(f"Difference: {abs(mean_qber - self.theoretical_p1):.4f}")
        
        # Statistical significance test
        self._perform_statistical_analysis(results)
        
        # Plot results
        self._plot_results(results)
        
        return results
    
    def _simulate_transmission(self, fixed_basis):
        """Simulate one transmission with fixed basis (no Eve)"""
        
        # Step 1: Alice prepares qubits
        alice_bits = [random.randint(0, 1) for _ in range(self.num_bits)]
        alice_bases = [fixed_basis] * self.num_bits  # All same basis
        
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
        
        # Step 2: Apply channel noise (no Eve eavesdropping)
        channel_noise_prob = np.random.normal(self.channel_noise_mean, self.channel_noise_std)
        channel_noise_prob = max(0, min(1, channel_noise_prob))  # Clamp to [0,1]
        
        for i in range(self.num_bits):
            if random.random() < channel_noise_prob:
                if fixed_basis == 0:  # Z basis - apply bit flip
                    circuits[i].x(0)
                else:  # X basis - apply phase flip
                    circuits[i].z(0)
        
        # Step 3: Bob measures (same basis as Alice)
        bob_bases = [fixed_basis] * self.num_bits
        bob_results = []
        
        for i in range(self.num_bits):
            qc = circuits[i].copy()
            
            # Apply Bob's measurement basis
            if bob_bases[i] == 1:  # X basis
                qc.h(0)
            
            qc.measure(0, 0)
            
            # Execute measurement
            job = self.simulator.run(qc, shots=1)
            result = job.result()
            counts = result.get_counts()
            bob_result = int(list(counts.keys())[0])
            bob_results.append(bob_result)
        
        # Step 4: Calculate QBER (all bases match, so all bits are sifted)
        errors = sum(1 for i in range(self.num_bits) if alice_bits[i] != bob_results[i])
        qber = errors / self.num_bits
        
        return qber
    
    def _perform_statistical_analysis(self, results):
        """Perform statistical tests to compare observed vs theoretical"""
        
        print(f"\n{'='*40}")
        print("STATISTICAL ANALYSIS")
        print(f"{'='*40}")
        
        for scenario_name, data in results.items():
            print(f"\n--- {scenario_name} ---")
            
            observed_mean = data['mean']
            theoretical = data['theoretical']
            observed_std = data['std']
            n = len(data['qber_values'])
            
            # One-sample t-test against theoretical value
            t_stat, p_value = stats.ttest_1samp(data['qber_values'], theoretical)
            
            # 95% confidence interval
            ci_margin = 1.96 * observed_std / math.sqrt(n)
            ci_lower = observed_mean - ci_margin
            ci_upper = observed_mean + ci_margin
            
            print(f"Observed: {observed_mean:.6f} ± {observed_std:.6f}")
            print(f"Theoretical: {theoretical:.6f}")
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
    
    def _plot_results(self, results):
        """Plot histogram of QBER results"""

        plt.figure(figsize=(10, 6))

        for scenario_name, data in results.items():
            # Histogram
            plt.hist(data['qber_values'], bins=50, alpha=0.7, density=True, 
                   label=f'Observed (n={len(data["qber_values"])})')
            
            # Theoretical value line
            plt.axvline(data['theoretical'], color='red', linestyle='--', linewidth=2,
                      label=f'Theoretical: {data["theoretical"]:.4f}')
            
            # Observed mean line
            plt.axvline(data['mean'], color='green', linestyle='-', linewidth=2,
                      label=f'Observed Mean: {data["mean"]:.4f}')
            
            plt.xlabel('QBER')
            plt.ylabel('Density')
            plt.title(f'{scenario_name}\nStd: {data["std"]:.4f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run the verification test"""
    
    # Test different parameter sets
    test_configs = [
        # {"noise_mean": 0.05, "noise_std": 0.1, "name": "Low Noise"},
        # {"noise_mean": 0.2, "noise_std": 0.15, "name": "Medium Noise"},
        {"noise_mean": 0.2, "noise_std": 0.02, "name": "High Noise"}
    ]
    
    for config in test_configs:
        print(f"\n{'#'*80}")
        print(f"TESTING CONFIGURATION: {config['name']}")
        print(f"{'#'*80}")
        
        # Create test instance
        test = FixedBasisQBERTest(
            num_bits=500,  # Fewer bits for faster testing
            channel_noise_mean=config['noise_mean'],
            channel_noise_std=config['noise_std']
        )
        
        # Run test for Z-basis
        print(f"\nTesting Z-basis...")
        results_z = test.run_fixed_basis_test(fixed_basis=0, num_runs=200)
        
        # Run test for X-basis  
        print(f"\nTesting X-basis...")
        results_x = test.run_fixed_basis_test(fixed_basis=1, num_runs=200)
        
        # Compare Z vs X basis results
        print(f"\n--- Z vs X Basis Comparison ---")
        for scenario in ["Channel Noise Only"]:
            print(f"{scenario}:")
            print(f"  Z-basis QBER: {results_z[scenario]['mean']:.4f}")
            print(f"  X-basis QBER: {results_x[scenario]['mean']:.4f}")
            print(f"  Theoretical: {test.theoretical_p1:.4f}")


if __name__ == "__main__":
    main()