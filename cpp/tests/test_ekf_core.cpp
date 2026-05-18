#include "rpo/ekf_core.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

int main() {
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    constexpr int N = 2;
    constexpr double dt = 0.1;
    constexpr int steps = 200;

    rpo::EkfCore ekf(N);
    VectorXd x0(N); x0 << 0.0, 0.0;
    MatrixXd P0 = MatrixXd::Identity(N, N) * 10.0;
    ekf.set_state(x0, P0);

    MatrixXd F(N, N); F << 1.0, dt, 0.0, 1.0;
    MatrixXd Q(N, N); Q << 1e-4, 0.0, 0.0, 1e-3;
    MatrixXd H(1, N); H << 1.0, 0.0;
    MatrixXd R(1, 1); R << 0.25;

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, std::sqrt(R(0, 0)));

    double truth_x = 0.0;
    const double truth_v = 1.5;

    for (int k = 0; k < steps; ++k) {
        truth_x += truth_v * dt;
        ekf.predict(F, Q);
        VectorXd y(1); y << (truth_x + noise(rng)) - (H * ekf.state())(0);
        ekf.update_linear(y, H, R);
    }

    const double err_x = std::abs(ekf.state()(0) - truth_x);
    const double err_v = std::abs(ekf.state()(1) - truth_v);
    std::printf("final |e_x|=%.4f m, |e_v|=%.4f m/s, trace(P)=%.4e\n",
                err_x, err_v, ekf.covariance().trace());
    assert(err_x < 0.5);
    assert(err_v < 0.1);
    assert(ekf.covariance().trace() < 1.0);
    std::printf("ekf_core smoke test PASSED\n");
    return 0;
}
