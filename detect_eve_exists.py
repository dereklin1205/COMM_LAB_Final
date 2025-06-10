import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import random
import matplotlib.pyplot as plt
from scipy import stats
import math

class SimpleEveDetection:
    def __init__(self, num_bits=1000, channel_noise_mean=0.05, channel_noise_std=0.1, fixed_basis=0):
        """
        Simplified Eve detection for q = 0.7 using fixed basis
        
        Args:
            num_bits: Number of qubits per test
            channel_noise_mean: Mean of channel noise p1
            channel_noise_std: Standard deviation of channel noise p1
            fixed_basis: 0 for Z-basis, 1 for X-basis
        """
        self.num_bits = num_bits
        self.channel_noise_mean = channel_noise_mean
        self.channel_noise_std = channel_noise_std
        self.fixed_basis = fixed_basis
        self.simulator = AerSimulator()
        self.q_value = 0.7  # Fixed q value
        
        # Calculate theoretical values
        self.theoretical_p1 = self._calculate_theoretical_p1()
        self.theoretical_qber_eve = self._calculate_series_error_prob(self.theoretical_p1, self.q_value)
        
        print(f"Fixed basis: {'Z-basis' if fixed_basis == 0 else 'X-basis'}")
        print(f"Theoretical p1 (no Eve): {self.theoretical_p1:.4f}")
        print(f"Theoretical QBER with Eve (q=0.7): {self.theoretical_qber_eve:.4f}")
        print(f"Expected QBER increase: {self.theoretical_qber_eve - self.theoretical_p1:.4f}")
    
    def _calculate_theoretical_p1(self):
        """Calculate theoretical p1 = P(Y <= X) where X~N(μ,σ²), Y~U(0,1)"""
        mu = self.channel_noise_mean
        sigma = self.channel_noise_std
        
        from scipy import integrate
        
        def integrand1(x):
            return x * stats.norm.pdf(x, mu, sigma)
        
        integral1, _ = integrate.quad(integrand1, 0, 1)
        integral2 = 1 - stats.norm.cdf(1, mu, sigma)
        
        return integral1 + integral2
    
    def _calculate_series_error_prob(self, p1, q):
        """Calculate series channel error: p1*(1-1/2q) + (1-p1)*1/2q"""
        series_error = p1 * (1 - (1/2)*q) + (1 - p1) * (1/2)*q
        return min(max(series_error, 0), 1)
    
    def _simulate_transmission(self, with_eve=False):
        """Simulate one transmission with fixed basis"""
        
        # Alice prepares random bits with fixed basis
        alice_bits = [random.randint(0, 1) for _ in range(self.num_bits)]
        
        circuits = []
        for i in range(self.num_bits):
            qc = QuantumCircuit(1, 1)
            
            # Encode Alice's bit
            if alice_bits[i] == 1:
                qc.x(0)
            
            # Apply fixed basis
            if self.fixed_basis == 1:  # X basis
                qc.h(0)
            
            circuits.append(qc)
        
        # Apply channel errors
        if with_eve:
            # With Eve: use series channel model
            channel_noise_prob = np.random.normal(self.channel_noise_mean, self.channel_noise_std)
            channel_noise_prob = max(0, min(1, channel_noise_prob))
            error_prob = self._calculate_series_error_prob(channel_noise_prob, self.q_value)
        else:
            # No Eve: only channel noise
            error_prob = np.random.normal(self.channel_noise_mean, self.channel_noise_std)
            error_prob = max(0, min(1, error_prob))
        
        # Apply errors
        for i in range(self.num_bits):
            if random.random() < error_prob:
                if self.fixed_basis == 0:  # Z basis - bit flip
                    circuits[i].x(0)
                else:  # X basis - phase flip
                    circuits[i].z(0)
        
        # Bob measures with same basis
        bob_results = []
        for i in range(self.num_bits):
            qc = circuits[i].copy()
            
            # Apply measurement basis
            if self.fixed_basis == 1:  # X basis
                qc.h(0)
            
            qc.measure(0, 0)
            
            # Execute
            job = self.simulator.run(qc, shots=1)
            result = job.result()
            counts = result.get_counts()
            bob_result = int(list(counts.keys())[0])
            bob_results.append(bob_result)
        
        # Calculate QBER
        errors = sum(1 for i in range(self.num_bits) if alice_bits[i] != bob_results[i])
        qber = errors / self.num_bits
        
        return qber
    
    def run_qber_test(self, num_trials=1000):
        """Run QBER test comparing scenarios with and without Eve"""
        
        print(f"\n{'='*60}")
        print(f"QBER TEST: EVE DETECTION (q = 0.7)")
        print(f"{'='*60}")
        print(f"Number of trials: {num_trials}")
        print(f"Bits per trial: {self.num_bits}")
        
        # Scenario 1: No Eve
        print(f"\n--- Running H0: No Eve scenario ---")
        qber_no_eve = []
        for i in range(num_trials):
            qber = self._simulate_transmission(with_eve=False)
            qber_no_eve.append(qber)
            
            if (i + 1) % 200 == 0:
                print(f"Completed {i + 1}/{num_trials} trials...")
        
        # Scenario 2: Eve present (q = 0.7)
        print(f"\n--- Running H1: Eve present (q = 0.7) scenario ---")
        qber_with_eve = []
        for i in range(num_trials):
            qber = self._simulate_transmission(with_eve=True)
            qber_with_eve.append(qber)
            
            if (i + 1) % 200 == 0:
                print(f"Completed {i + 1}/{num_trials} trials...")
        
        # Statistical analysis
        self._analyze_results(qber_no_eve, qber_with_eve)
        
        # Plot results
        self._plot_results(qber_no_eve, qber_with_eve)
        
        return qber_no_eve, qber_with_eve
    
    def _analyze_results(self, qber_no_eve, qber_with_eve):
        """Analyze and compare QBER results"""
        
        # Calculate statistics
        mean_no_eve = np.mean(qber_no_eve)
        std_no_eve = np.std(qber_no_eve)
        mean_with_eve = np.mean(qber_with_eve)
        std_with_eve = np.std(qber_with_eve)
        
        print(f"\n{'='*50}")
        print("QBER ANALYSIS RESULTS")
        print(f"{'='*50}")
        
        print(f"No Eve (H0):")
        print(f"  Observed QBER: {mean_no_eve:.4f} ± {std_no_eve:.4f}")
        print(f"  Theoretical: {self.theoretical_p1:.4f}")
        print(f"  Difference: {abs(mean_no_eve - self.theoretical_p1):.4f}")
        
        print(f"\nEve Present (H1, q=0.7):")
        print(f"  Observed QBER: {mean_with_eve:.4f} ± {std_with_eve:.4f}")
        print(f"  Theoretical: {self.theoretical_qber_eve:.4f}")
        print(f"  Difference: {abs(mean_with_eve - self.theoretical_qber_eve):.4f}")
        
        # print(f"\nComparison:")
        # qber_increase = mean_with_eve - mean_no_eve
        # print(f"  QBER increase due to Eve: {qber_increase:.4f}")
        # print(f"  Theoretical increase: {self.theoretical_qber_eve - self.theoretical_p1:.4f}")
        
        # Statistical test for Eve detection
        # t_stat, p_value = stats.ttest_ind(qber_with_eve, qber_no_eve, equal_var=False)
        # p_value_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        
        # # Effect size
        # pooled_std = np.sqrt((np.var(qber_with_eve) + np.var(qber_no_eve)) / 2)
        # cohens_d = (mean_with_eve - mean_no_eve) / pooled_std
        
        # print(f"\nStatistical Test (Eve Detection):")
        # print(f"  t-statistic: {t_stat:.4f}")
        # print(f"  p-value (one-sided): {p_value_one_sided:.6f}")
        # print(f"  Effect size (Cohen's d): {cohens_d:.4f}")
        
        # # Decision
        # alpha = 0.05
        # if p_value_one_sided < alpha:
        #     decision = "🚨 EVE DETECTED"
        #     confidence = (1 - p_value_one_sided) * 100
        # else:
        #     decision = "✅ No strong evidence of Eve"
        #     confidence = (1 - p_value_one_sided) * 100
        
        # print(f"  Decision (α = 0.05): {decision}")
        # print(f"  Confidence: {confidence:.2f}%")
        
        # # Detection threshold
        # threshold = mean_no_eve + 1.96 * std_no_eve  # 95% confidence
        # print(f"\nDetection Threshold (95% confidence): {threshold:.4f}")
        # if mean_with_eve > threshold:
        #     print(f"  ✅ Eve scenario exceeds threshold - Detectable!")
        # else:
        #     print(f"  ❌ Eve scenario below threshold - Not reliably detectable")
    
    def _plot_results(self, qber_no_eve, qber_with_eve):
        """Plot QBER comparison results"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: QBER distributions
        ax1.hist(qber_no_eve, bins=30, alpha=0.7, density=True, 
                label='No Eve (H0)', color='blue')
        ax1.hist(qber_with_eve, bins=30, alpha=0.7, density=True, 
                label='Eve Present (H1)', color='red')
        
        # Add theoretical lines
        ax1.axvline(self.theoretical_p1, color='blue', linestyle='--', 
                   label=f'Theoretical (No Eve): {self.theoretical_p1:.4f}')
        ax1.axvline(self.theoretical_qber_eve, color='red', linestyle='--', 
                   label=f'Theoretical (Eve): {self.theoretical_qber_eve:.4f}')
        
        ax1.set_xlabel('QBER')
        ax1.set_ylabel('Density')
        ax1.set_title(f'QBER Distributions - {"Z" if self.fixed_basis == 0 else "X"}-basis (q=0.7)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Box plot comparison
        ax2.boxplot([qber_no_eve, qber_with_eve], 
                   labels=['No Eve', 'Eve (q=0.7)'])
        ax2.set_ylabel('QBER')
        ax2.set_title('QBER Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mean comparison with error bars
        scenarios = ['No Eve', 'Eve (q=0.7)']
        means = [np.mean(qber_no_eve), np.mean(qber_with_eve)]
        stds = [np.std(qber_no_eve), np.std(qber_with_eve)]
        theoretical = [self.theoretical_p1, self.theoretical_qber_eve]
        
        x_pos = [0, 1]
        ax3.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
               color=['blue', 'red'], label='Observed')
        ax3.scatter(x_pos, theoretical, color='black', s=100, 
                   label='Theoretical', zorder=5)
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(scenarios)
        ax3.set_ylabel('Mean QBER')
        ax3.set_title('Mean QBER Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: QBER difference
        qber_diff = np.array(qber_with_eve) - np.array(qber_no_eve)
        ax4.hist(qber_diff, bins=30, alpha=0.7, color='green')
        ax4.axvline(0, color='black', linestyle='-', label='No difference')
        ax4.axvline(np.mean(qber_diff), color='red', linestyle='--', 
                   label=f'Mean diff: {np.mean(qber_diff):.4f}')
        ax4.axvline(self.theoretical_qber_eve - self.theoretical_p1, 
                   color='blue', linestyle=':', 
                   label=f'Theoretical diff: {self.theoretical_qber_eve - self.theoretical_p1:.4f}')
        
        ax4.set_xlabel('QBER Difference (Eve - No Eve)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('QBER Increase Due to Eve')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run simplified Eve detection test"""
    
    print("SIMPLIFIED EVE DETECTION TEST (q = 0.7)")
    print("="*50)
    
    # # Test with Z-basis
    
    
    # Test with X-basis
    print("\n" + "="*60)
    print("\nTesting with X-basis:")
    detector_x = SimpleEveDetection(
        num_bits=1000,
        channel_noise_mean=0.2,
        channel_noise_std=0.02,
        fixed_basis=1  # X-basis
    )
    
    qber_no_eve_x, qber_with_eve_x = detector_x.run_qber_test(num_trials=1000)
    print("\nTesting with Z-basis:")
    # detector_z = SimpleEveDetection(
    #     num_bits=1000,
    #     channel_noise_mean=0.05,
    #     channel_noise_std=0.1,
    #     fixed_basis=0  # Z-basis
    # )
    
    # qber_no_eve_z, qber_with_eve_z = detector_z.run_qber_test(num_trials=1000)
    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY: Z-BASIS vs X-BASIS")
    print(f"{'='*60}")
    
    # z_increase = np.mean(qber_with_eve_z) - np.mean(qber_no_eve_z)
    x_increase = np.mean(qber_with_eve_x) - np.mean(qber_no_eve_x)
    
    # print(f"Z-basis QBER increase: {z_increase:.4f}")
    print(f"X-basis QBER increase: {x_increase:.4f}")
    # print(f"Better detection basis: {'Z-basis' if z_increase > x_increase else 'X-basis'}")


if __name__ == "__main__":
    main()