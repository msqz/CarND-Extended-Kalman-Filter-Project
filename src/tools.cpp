#include <iostream>
#include <math.h>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse = VectorXd(4);
  rmse << 0, 0, 0, 0;

  if (estimations.size() == 0 || estimations.size() != ground_truth.size()) {
    std::cout << "Invalid input size" << std::endl;
    return rmse;
  }

  for (int i = 0; i < estimations.size(); i++) {
    VectorXd residual = estimations[i] - ground_truth[i];
    rmse = rmse.array() + residual.array().pow(2);
  }
  rmse /= estimations.size();
  rmse = rmse.array().sqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj = MatrixXd(3,4);
  Hj << 0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0;
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  double px2_plus_py2 = pow(px, 2) + pow(py, 2);
  if (px2_plus_py2 < 0.0001) {
    std::cout << "Sum of squared position coeff. equals 0" << std::endl;
    return Hj;
  }

  Hj(0,0) = px / sqrt(px2_plus_py2);
  Hj(0,1) = py / sqrt(px2_plus_py2);
  Hj(0,2) = 0;
  Hj(0,3) = 0;
  Hj(1,0) = -py / px2_plus_py2;
  Hj(1,1) = px / px2_plus_py2;
  Hj(1,2) = 0;
  Hj(1,3) = 0;
  Hj(2,0) = py*(vx*py - vy*px) / pow(px2_plus_py2, 3/2.0);
  Hj(2,1) = px*(vy*px - vx*py) / pow(px2_plus_py2, 3/2.0);
  Hj(2,2) = px / sqrt(px2_plus_py2);
  Hj(2,3) = py / sqrt(px2_plus_py2);

  return Hj;
}
