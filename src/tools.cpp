#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0, 0, 0, 0;

	// check the validity of inputs:
	if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
		std::cout << "Invalid estimation or ground_truth data" << std::endl;
		return rmse;
	}

	// accumulate squared residuals
	for (int i = 0; i < (int)estimations.size(); ++i) {
		rmse = rmse.array() + (estimations[i] - ground_truth[i]).array()*(estimations[i] - ground_truth[i]).array();
	}

	// calculate the mean
	rmse = rmse / estimations.size();
	// calculate the squared root
	rmse = rmse.array().sqrt();

	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	MatrixXd Hj =MatrixXd::Zero(3,4);
	// recover state parameters
	double px = x_state(0);
	double py = x_state(1);
	double vx = x_state(2);
	double vy = x_state(3);


	// check division by zero
	if (fabs(px * px + py * py) < 0.0001) {
		std::cout << "CalculateJacobian () - Error - Division by Zero" <<std::endl;
		return Hj;
	}
	// compute the Jacobian matrix
	Hj << px / (sqrt(px*px + py * py)), py / (sqrt(px*px + py * py)), 0, 0,
		 -py / (px*px + py * py), px / (px*px + py * py), 0, 0,
		 (py*(vx*py - vy * px)) / (pow(px*px + py * py, 1.5)), (px*(vy*px - vx * py)) / (pow(px*px + py * py, 1.5)), px / (sqrt(px*px + py * py)), py / (sqrt(px*px + py * py));

	return Hj;
}
