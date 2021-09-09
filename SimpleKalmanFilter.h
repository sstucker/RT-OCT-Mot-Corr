#include <Eigen/Dense>

#pragma once
class SimpleKalmanFilter
{

public:

	SimpleKalmanFilter() { }

	// w/o control matrix
	SimpleKalmanFilter(
		const Eigen::MatrixXd& A,  // State evolution matrix
		const Eigen::MatrixXd& H,  // Measurement matrix
		const Eigen::MatrixXd& Q,  // Process uncertainty matrix
		const Eigen::MatrixXd& R,  // Sensor noise uncertainity matrix 
		const Eigen::VectorXd& X0, // Init state
		const Eigen::MatrixXd& P0  // Init cov
	) :
		A(A), H(H), Q(Q), R(R), X(X0), P(P0)
	{
		I = Eigen::MatrixXd::Identity(A.rows(), A.rows());
	}

	Eigen::VectorXd observeAndPredict(const Eigen::VectorXd& y)
	{
		update(y);
		predict();
		return this->X;
	}

	void setQ(const Eigen::MatrixXd& Q)
	{
		this->Q = Q;
	}

	void setR(const Eigen::MatrixXd& R)
	{
		this->R = R;
	}

	Eigen::VectorXd getState() { return X; }

protected:

	Eigen::MatrixXd A, B, H, Q, R;
	Eigen::MatrixXd P;  // Covariance
	Eigen::MatrixXd K;  // Kalman gain
	Eigen::MatrixXd I;  // Identity matrix

	Eigen::VectorXd X;  // State

	void update(const Eigen::VectorXd& y)
	{
		K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
		X += K * (y - H * X);
		P = (I - K * H) * P;
	}

	void predict()
	{
		X = A * X;
		P = A * P * A.transpose() + Q;
	}

};

