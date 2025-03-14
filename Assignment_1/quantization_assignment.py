"""

Assignment: Quantization
Contributors:
    - Amir Kedis
    - Akram Hany

"""


# =================================================================
# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
# =================================================================


# =================================================================
# QUANTIZER
def uniform_quantizer(in_val: np.ndarray, n_bits: int, xmax: float, m: int) -> np.ndarray:
    """Uniform quantizer for given input values."""
    # Calculate number of levels and step size
    levels = 2**n_bits
    delta = 2 * xmax / levels

    # Convert input to numpy array if it's not already
    in_val = np.asarray(in_val)

    # Clip input to the valid range
    in_val_clipped = np.clip(in_val, -xmax + delta / 2, xmax - delta / 2)

    # Shift and scale based on quantizer type
    if m == 0:  # Midrise quantizer
        q_ind = np.floor((in_val_clipped + xmax) / delta).astype(int)
    else:  # Midtread quantizer (m == 1)
        q_ind = np.floor((in_val_clipped + xmax - delta / 2) / delta).astype(int)

    # Ensure indices are within valid range
    q_ind = np.clip(q_ind, 0, levels - 1)

    return q_ind


def uniform_dequantizer(q_ind: np.ndarray, n_bits: int, xmax: float, m: int) -> np.ndarray:
    """Uniform dequantizer for given quantized indices."""
    # Calculate number of levels and step size
    levels = 2**n_bits
    delta = 2 * xmax / levels

    # Convert indices to numpy array if they're not already
    q_ind = np.asarray(q_ind)

    # Ensure indices are within valid range
    q_ind = np.clip(q_ind, 0, levels - 1)

    # Apply dequantization based on quantizer type
    if m == 0:  # Midrise quantizer
        deq_val = (q_ind + 0.5) * delta - xmax
    else:  # Midtread quantizer (m == 1)
        deq_val = q_ind * delta - xmax + delta / 2

    return deq_val
# =================================================================


# =================================================================
# TEST
def test_deterministic_input():
    """Test quantizer/dequantizer with a deterministic ramp input"""
    # Generate ramp signal
    x = np.arange(-6, 6.01, 0.01)  # Include 6 in the range
    n_bits = 3
    xmax = 6

    # Test for m = 0 (midrise)
    m = 0
    q_ind = uniform_quantizer(x, n_bits, xmax, m)
    y_midrise = uniform_dequantizer(q_ind, n_bits, xmax, m)

    # Plot results for m = 0
    plt.figure(figsize=(10, 6))
    plt.plot(x, x, "b-", linewidth=1.5, label="Original Signal")
    plt.plot(x, y_midrise, "r-", linewidth=1.5, label="Quantized Signal")
    plt.title("Uniform Quantization: Midrise (m = 0)")
    plt.xlabel("Input Value")
    plt.ylabel("Output Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("midrise_quantization.png")
    plt.show()

    # Test for m = 1 (midtread)
    m = 1
    q_ind = uniform_quantizer(x, n_bits, xmax, m)
    y_midtread = uniform_dequantizer(q_ind, n_bits, xmax, m)

    # Plot results for m = 1
    plt.figure(figsize=(10, 6))
    plt.plot(x, x, "b-", linewidth=1.5, label="Original Signal")
    plt.plot(x, y_midtread, "r-", linewidth=1.5, label="Quantized Signal")
    plt.title("Uniform Quantization: Midtread (m = 1)")
    plt.xlabel("Input Value")
    plt.ylabel("Output Value")
    plt.legend()
    plt.grid(True)
    plt.savefig("midtread_quantization.png")
    plt.show()


def test_uniform_random_input():
    """Test with uniform random input and calculate SNR"""
    # Generate random uniform samples
    np.random.seed(42)  # For reproducibility
    num_samples = 10000
    x_unif = -5 + 10 * np.random.rand(num_samples)  # Uniform between -5 and 5
    xmax = 5
    m = 0

    # Arrays to store SNR values
    bits_range = np.arange(2, 9)
    snr_sim = np.zeros_like(bits_range, dtype=float)
    snr_theory = np.zeros_like(bits_range, dtype=float)

    # Loop over different bit rates
    for i, n_bits in enumerate(bits_range):
        # Quantize and dequantize
        q_ind = uniform_quantizer(x_unif, n_bits, xmax, m)
        y = uniform_dequantizer(q_ind, n_bits, xmax, m)

        # Calculate error and SNR
        error = x_unif - y
        signal_power = np.mean(x_unif**2)
        noise_power = np.mean(error**2)
        snr_sim[i] = 10 * np.log10(signal_power / noise_power)

        # Theoretical SNR for uniform distribution: 6.02*n_bits + 1.76 dB
        snr_theory[i] = 6.02 * n_bits + 1.76

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(bits_range, snr_sim, "bo-", linewidth=1.5, label="Simulated SNR")
    plt.plot(bits_range, snr_theory, "r--", linewidth=1.5, label="Theoretical SNR")
    plt.title("SNR vs. Number of Bits for Uniform Distribution")
    plt.xlabel("Number of Bits")
    plt.ylabel("SNR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig("uniform_snr.png")
    plt.show()

    return bits_range, snr_sim


def test_nonuniform_random_input():
    """Test with non-uniform random input (exponential distribution)"""
    # Generate exponential random samples with random polarity
    np.random.seed(42)  # For reproducibility
    num_samples = 10000
    mag = np.random.exponential(
        scale=1.0, size=num_samples
    )  # Exponential distribution with mean 1
    polarity = np.sign(np.random.rand(num_samples) - 0.5)  # Random +1 or -1
    x_exp = polarity * mag

    # Set appropriate xmax to cover most of the distribution
    xmax = 5  # Adjust based on distribution range
    m = 0

    # Arrays to store SNR values
    bits_range = np.arange(2, 9)
    snr_exp = np.zeros_like(bits_range, dtype=float)

    # Loop over different bit rates
    for i, n_bits in enumerate(bits_range):
        # Quantize and dequantize
        q_ind = uniform_quantizer(x_exp, n_bits, xmax, m)
        y = uniform_dequantizer(q_ind, n_bits, xmax, m)

        # Calculate error and SNR
        error = x_exp - y
        signal_power = np.mean(x_exp**2)
        noise_power = np.mean(error**2)
        snr_exp[i] = 10 * np.log10(signal_power / noise_power)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(bits_range, snr_exp, "go-", linewidth=1.5, label="Uniform Quantizer")
    plt.title("SNR vs. Number of Bits for Exponential Distribution")
    plt.xlabel("Number of Bits")
    plt.ylabel("SNR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig("nonuniform_snr.png")
    plt.show()

    # Return for comparison with μ-law quantization
    return bits_range, snr_exp


def test_mu_law_quantization():
    """Test μ-law quantization for non-uniform input"""
    # Generate exponential random samples with random polarity
    np.random.seed(42)  # For reproducibility
    num_samples = 10000
    mag = np.random.exponential(
        scale=1.0, size=num_samples
    )  # Exponential distribution with mean 1
    polarity = np.sign(np.random.rand(num_samples) - 0.5)  # Random +1 or -1
    x_exp = polarity * mag

    # Set appropriate xmax to cover most of the distribution
    xmax = 5  # Adjust based on distribution range
    m = 0

    # μ values to test
    mu_values = [0, 5, 100, 200]
    bits_range = np.arange(2, 9)

    # Arrays to store SNR values for different μ values
    snr_mu = np.zeros((len(mu_values), len(bits_range)))

    # Loop over different μ values
    for mu_idx, mu in enumerate(mu_values):
        # Loop over different bit rates
        for bit_idx, n_bits in enumerate(bits_range):
            if mu == 0:
                # For μ = 0, use uniform quantization directly
                q_ind = uniform_quantizer(x_exp, n_bits, xmax, m)
                y = uniform_dequantizer(q_ind, n_bits, xmax, m)
            else:
                # Apply μ-law expansion before quantization
                x_expanded = (
                    np.sign(x_exp) * np.log(1 + mu * np.abs(x_exp)) / np.log(1 + mu)
                )

                # Scale expanded signal to [-xmax, xmax]
                scale_factor = xmax
                x_expanded_scaled = x_expanded * scale_factor

                # Quantize expanded signal
                q_ind = uniform_quantizer(x_expanded_scaled, n_bits, xmax, m)
                y_expanded = uniform_dequantizer(q_ind, n_bits, xmax, m)

                # Apply μ-law compression to recover signal
                y_expanded_scaled = y_expanded / scale_factor
                y = (
                    np.sign(y_expanded_scaled)
                    * (1 / mu)
                    * ((1 + mu) ** np.abs(y_expanded_scaled) - 1)
                )

            # Calculate error and SNR
            error = x_exp - y
            signal_power = np.mean(x_exp**2)
            noise_power = np.mean(error**2)
            snr_mu[mu_idx, bit_idx] = 10 * np.log10(signal_power / noise_power)

    # Plot results
    plt.figure(figsize=(10, 6))
    for mu_idx, mu in enumerate(mu_values):
        label = "μ = " + str(mu)
        if mu == 0:
            label += " (Uniform)"
        plt.plot(bits_range, snr_mu[mu_idx, :], "o-", linewidth=1.5, label=label)

    plt.title("SNR vs. Number of Bits for Different μ Values")
    plt.xlabel("Number of Bits")
    plt.ylabel("SNR (dB)")
    plt.legend()
    plt.grid(True)
    plt.savefig("mu_law_comparison.png")
    plt.show()
# =================================================================


# =================================================================
# MAIN
def main():
    """Run all tests for the quantization assignment"""
    # Part 3: Test with deterministic input
    print("Testing with deterministic input...")
    test_deterministic_input()

    # Part 4: Test with uniform random input
    print("Testing with uniform random input...")
    test_uniform_random_input()

    # Part 5: Test with non-uniform random input
    print("Testing with non-uniform random input...")
    test_nonuniform_random_input()

    # Part 6: Test with μ-law quantization
    print("Testing with μ-law quantization...")
    test_mu_law_quantization()


if __name__ == "__main__":
    main()

# =================================================================