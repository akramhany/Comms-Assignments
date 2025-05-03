import matplotlib.pyplot as plt
import numpy as np

NUMBER_SAMPLES = 100  # Set the number of samples


def gm_bases(s1, s2):
    # applying 2 steps of Gram-Schmidt process
    norm_s1 = np.linalg.norm(s1)
    phi1 = s1 / norm_s1

    proj = np.dot(s2, phi1) * phi1
    ortho = s2 - proj

    norm_ortho = np.linalg.norm(ortho)
    phi2 = ortho / norm_ortho if norm_ortho != 0 else np.zeros_like(s1)

    return phi1, phi2


def signal_space(s, phi1, phi2):
    # NOTE: Simply project into phi1, phi2 axes (linear algebra)
    v1 = np.dot(s, phi1) / NUMBER_SAMPLES
    v2 = np.dot(s, phi2) / NUMBER_SAMPLES
    return v1, v2


def add_awgn(signal, sigma2):
    # NOTE: awgn: Additive White Gaussian Noise
    noise = np.random.normal(0, np.sqrt(sigma2), NUMBER_SAMPLES)
    return signal + noise


def plot_signal(t, signal, title, filename):
    # Simple utility to not repeat this 4 times below :)
    plt.figure()
    plt.plot(t, signal, linewidth=2)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def run_awgn_experiment(
    s1, s2, phi1, phi2, num_samples=NUMBER_SAMPLES, eb_sigma2_db_list=[-5, 0, 10, 15]
):
    # NOTE: s1, s2 are the signals, phi1, phi2 are the basis functions
    # NOTE: eb_sigma2_db_list is the list of SNR values in dB (E_b/N)
    Es = np.sum(s1**2)
    Es /= np.sqrt(NUMBER_SAMPLES)
    v1_s1, v2_s1 = signal_space(s1, phi1, phi2)
    v1_s2, v2_s2 = signal_space(s2, phi1, phi2)
    results = {}

    for db in eb_sigma2_db_list:
        sigma2 = Es / (10 ** (db / 10))
        r1_points = []
        r2_points = []

        for _ in range(num_samples):
            r1 = add_awgn(s1, sigma2)
            r2 = add_awgn(s2, sigma2)
            v1_r1, v2_r1 = signal_space(r1, phi1, phi2)
            v1_r2, v2_r2 = signal_space(r2, phi1, phi2)
            r1_points.append((v1_r1, v2_r1))
            r2_points.append((v1_r2, v2_r2))

        results[db] = (np.array(r1_points), np.array(r2_points))

        # Plot
        plt.figure()
        # Plotting the noisy signal space
        plt.scatter(*zip(*r1_points), label="r1 (s1 + noise)", alpha=0.4, color="blue")
        plt.scatter(*zip(*r2_points), label="r2 (s2 + noise)", alpha=0.4, color="red")
        plt.plot(v1_s1, v2_s1, "o", label="s1 ideal", markersize=10, color="indigo")
        plt.plot(v1_s2, v2_s2, "o", label="s2 ideal", markersize=10, color="crimson")

        plt.xlabel("v1")
        plt.ylabel("v2")
        plt.title(f"AWGN Signal Space ($E/\\sigma^2$ = {db} dB)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"noisy_signal_space_{db}dB.png")
        plt.close()

    return results


if __name__ == "__main__":
    # NOTE: our time space 1 is 100 samples for simplicity
    t = np.linspace(0, 1, NUMBER_SAMPLES)

    # Fix the scale: divide by sqrt(N) to normalize energy
    scale = np.sqrt(NUMBER_SAMPLES)

    s1 = np.ones(NUMBER_SAMPLES)
    s2 = np.concatenate(
        [np.ones(int(0.75 * NUMBER_SAMPLES)), -1 * np.ones(int(0.25 * NUMBER_SAMPLES))]
    )

    plot_signal(t, s1, "Signal s1(t)", "signal_s1.png")
    plot_signal(t, s2, "Signal s2(t)", "signal_s2.png")

    phi1, phi2 = gm_bases(s1, s2)
    phi1 = phi1 * scale
    phi2 = phi2 * scale
    plot_signal(t, phi1, "Basis Function φ1(t)", "basis_phi1.png")
    plot_signal(t, phi2, "Basis Function φ2(t)", "basis_phi2.png")

    # Signal space representation
    v1_s1, v2_s1 = signal_space(s1, phi1, phi2)
    v1_s2, v2_s2 = signal_space(s2, phi1, phi2)

    plt.figure()
    plt.plot(v1_s1, v2_s1, "o", label="s1", markersize=10, color="indigo")
    plt.plot(v1_s2, v2_s2, "o", label="s2", markersize=10, color="crimson")
    plt.plot([0, v1_s1], [0, v2_s1], "-", color="indigo", label="Vector to s1")
    plt.plot([0, v1_s2], [0, v2_s2], "-", color="crimson", label="Vector to s2")
    plt.xlabel("v1")
    plt.ylabel("v2")
    plt.title("Signal Space Representation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("signal_space_clean.png")
    plt.close()

    run_awgn_experiment(s1, s2, phi1, phi2)
