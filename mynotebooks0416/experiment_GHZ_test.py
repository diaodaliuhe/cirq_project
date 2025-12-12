import cirq
import numpy as np
from cirq.noise import (
    IdleNoiseModel,
    RyGateNoiseModel,
    PhotonDecayNoiseModel,
    SampleFluxNoiseModel,
    AverageFluxNoiseModel,
    FluxNoiseContext,
    CompositeNoiseModel,
)

# 构建电路
q0, q1, q2 = cirq.LineQubit.range(3)
circuit = cirq.Circuit(
    cirq.H(q0),
    cirq.CX(q0, q1),
    cirq.CX(q1, q2)
)
circuit_no_measure = circuit  # 没有测量门

# 理想态
sim_ideal = cirq.DensityMatrixSimulator()
rho_ideal = sim_ideal.simulate(circuit_no_measure).final_density_matrix

# 固定噪声模型（除了flux noise）
fixed_models = [
    PhotonDecayNoiseModel(),
    RyGateNoiseModel(),
    IdleNoiseModel()
]

# 采样次数
n_samples = 100
fidelities_sample_vs_std = []
fidelities_sample_vs_channel = []
rho_samples = []

# 平均通道
model_flux_avg = AverageFluxNoiseModel()
composite_model_avg = CompositeNoiseModel(fixed_models + [model_flux_avg])
sim_avg = cirq.DensityMatrixSimulator(noise=composite_model_avg)
rho_channel = sim_avg.simulate(circuit_no_measure).final_density_matrix

for i in range(n_samples):
    # 每次新的 flux noise context + model
    context = FluxNoiseContext(phi_rms=0.01, qubits=[q0, q1, q2])
    context.generate()
    model_flux_sample = SampleFluxNoiseModel(context)

    composite_model_sample = CompositeNoiseModel(fixed_models + [model_flux_sample])
    sim_sample = cirq.DensityMatrixSimulator(noise=composite_model_sample)
    rho_sample = sim_sample.simulate(circuit_no_measure).final_density_matrix
    rho_samples.append(rho_sample)

    f_std = cirq.fidelity(rho_ideal, rho_sample, qid_shape=(2, 2, 2))
    f_channel = cirq.fidelity(rho_channel, rho_sample, qid_shape=(2, 2, 2))

    fidelities_sample_vs_std.append(f_std)
    fidelities_sample_vs_channel.append(f_channel)

# 平均 sample fidelity
rho_avg_sample = np.mean(rho_samples, axis=0)

# 最终统计
f_channel_vs_std = cirq.fidelity(rho_ideal, rho_channel, qid_shape=(2, 2, 2))
f_avg_sample_vs_channel = cirq.fidelity(rho_channel, rho_avg_sample, qid_shape=(2, 2, 2))

# 输出文件
with open("fidelity_report.txt", "w") as f:
    f.write(f"{'Index':<6} {'Fid(sampled,std)':<20} {'Fid(sampled,channel)':<20}\n")
    for i, (fs, fc) in enumerate(zip(fidelities_sample_vs_std, fidelities_sample_vs_channel), 1):
        f.write(f"{i:<6} {fs:<20.6f} {fc:<20.6f}\n")
    f.write("\n")
    f.write(f"Fidelity(channel, standard): {f_channel_vs_std:.6f}\n")
    f.write(f"Fidelity(avg(sampled), channel): {f_avg_sample_vs_channel:.6f}\n")

print("Fidelity report written to fidelity_report.txt")