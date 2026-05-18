#pragma once

#include <Eigen/Dense>
#include <stdexcept>

namespace rpo {

class EkfCore {
public:
    using Vec = Eigen::VectorXd;
    using Mat = Eigen::MatrixXd;

    explicit EkfCore(int n_state)
        : n_(n_state),
          x_(Vec::Zero(n_state)),
          P_(Mat::Identity(n_state, n_state)),
          I_(Mat::Identity(n_state, n_state))
    {
        if (n_state <= 0) throw std::invalid_argument("n_state must be > 0");
    }

    void set_state(const Vec& x, const Mat& P) {
        if (x.size() != n_) throw std::invalid_argument("state size mismatch");
        if (P.rows() != n_ || P.cols() != n_)
            throw std::invalid_argument("covariance shape mismatch");
        x_ = x;
        P_ = sym(P);
    }

    const Vec& state()      const noexcept { return x_; }
    const Mat& covariance() const noexcept { return P_; }
    int        dim()        const noexcept { return n_; }

    void predict(const Mat& F, const Mat& Q) {
        square_or_throw(F, "F");
        square_or_throw(Q, "Q");
        x_ = F * x_;
        P_ = sym(F * P_ * F.transpose() + Q);
    }

    Mat update_linear(const Vec& y, const Mat& H, const Mat& R) {
        if (H.cols() != n_)         throw std::invalid_argument("H cols mismatch");
        if (H.rows() != y.size())   throw std::invalid_argument("H rows / y mismatch");
        if (R.rows() != y.size() || R.cols() != y.size())
            throw std::invalid_argument("R dimensions mismatch");

        Mat S   = H * P_ * H.transpose() + R;
        Mat PHt = P_ * H.transpose();

        Mat K   = S.transpose().ldlt().solve(PHt.transpose()).transpose();

        x_ += K * y;
        Mat IKH = I_ - K * H;
        P_ = sym(IKH * P_ * IKH.transpose() + K * R * K.transpose());
        return K;
    }

private:
    int n_;
    Vec x_;
    Mat P_;
    Mat I_;

    static Mat sym(const Mat& A) { return 0.5 * (A + A.transpose()); }

    void square_or_throw(const Mat& M, const char* name) const {
        if (M.rows() != n_ || M.cols() != n_)
            throw std::invalid_argument(std::string(name) + " must be n_state x n_state");
    }
};

}
