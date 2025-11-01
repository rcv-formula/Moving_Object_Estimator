// ekf_module.hpp
#pragma once
/********************************************************************************
* Unified EKF single-header module
* - Contains: FixedSizeQueue, MeasurementPackage, Tools, KalmanFilter, FusionEKF
* - Keeps original class names so it’s drop-in friendly.
*********************************************************************************/

#include <deque>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <cstdint>

#include "Eigen/Dense"

// ----------------------------- FixedSizeQueue ---------------------------------
#ifndef FIXED_SIZE_QUEUE
#define FIXED_SIZE_QUEUE
#define SAVE_FRAME_NUM 2

template <typename T>
class FixedSizeQueue {
private:
    std::deque<T> data;
    const size_t maxSize = SAVE_FRAME_NUM;

public:
    FixedSizeQueue() {}

    // 원본 그대로 유지 (필요 시 const T& 로 바꿔도 무방)
    void push(T& value) {
        if (data.size() >= maxSize) {
            data.pop_front();
        }
        data.push_back(value);
    }

    void print() {
        std::cout << "최근 " << data.size() << "개 데이터: ";
        for (const auto& elem : data) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }

    const std::deque<T>& getData() const {
        return data;
    }

    int getMaxSize(){
        return static_cast<int>(maxSize);
    }
};
#endif // FIXED_SIZE_QUEUE

// --------------------------- MeasurementPackage -------------------------------
#ifndef MEASUREMENT_PACKAGE_H_
#define MEASUREMENT_PACKAGE_H_
struct MeasurementPackage {
  // 센서 타입(필요시 확장 가능). 여기서는 LASER만 사용.
  enum SensorType {
    LASER = 0
  } sensor_type_ = LASER;

  Eigen::VectorXd raw_measurements_;   // z = [x, y]^T
  std::int64_t    timestamp_ = 0;      // ns 단위 권장
};
#endif // MEASUREMENT_PACKAGE_H_

// ---------------------------------- Tools -------------------------------------
#ifndef TOOLS_H_
#define TOOLS_H_
class Tools {
public:
  Tools() = default;
  ~Tools() = default;

  // 필요 시 RMSE 등 유틸 추가 가능. 여기서는 자리만 유지.
  static Eigen::VectorXd CalculateRMSE(
      const std::vector<Eigen::VectorXd> &estimations,
      const std::vector<Eigen::VectorXd> &ground_truth) {
    Eigen::VectorXd rmse(4);
    rmse.setZero();
    if (estimations.empty() || estimations.size() != ground_truth.size()) {
      return rmse;
    }
    for (size_t i = 0; i < estimations.size(); ++i) {
      Eigen::VectorXd r = estimations[i] - ground_truth[i];
      rmse += r.array().square().matrix();
    }
    rmse /= static_cast<double>(estimations.size());
    rmse = rmse.array().sqrt().matrix();
    return rmse;
  }
};
#endif // TOOLS_H_

// -------------------------------- KalmanFilter --------------------------------
#ifndef KALMAN_FILTER_H_
#define KALMAN_FILTER_H_

using Eigen::MatrixXd;
using Eigen::VectorXd;

class KalmanFilter {
public:
  const float PI2 = 2 * static_cast<float>(M_PI);

  FixedSizeQueue<int> fixed_queue;

  int is_dynamic = 0;               // 0:NONE, 1:dynamic, 2:static
  std::vector<std::string> status;  // ["", "dynamic", "static"]

  // State
  Eigen::VectorXd x_;  // [x, y, vx, vy]^T

  // Covariances & Matrices
  Eigen::MatrixXd P_;  // state covariance
  Eigen::MatrixXd F_;  // state transition
  Eigen::MatrixXd Q_;  // process covariance
  Eigen::MatrixXd H_;  // measurement model
  Eigen::MatrixXd R_;  // measurement covariance

  KalmanFilter() {
    is_dynamic = false;
    status.resize(3);
    status[0] = "";
    status[1] = "dynamic";
    status[2] = "static";
  }

  virtual ~KalmanFilter() {}

  // 원 코드명 유지
  void InitInit(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
  }

  void Predict() {
    x_ = F_ * x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_ * P_ * Ft + Q_;
  }

  void Update(const VectorXd &z) {
    const VectorXd z_pred = H_ * x_;
    const VectorXd y = z - z_pred;
    const MatrixXd Ht = H_.transpose();
    const MatrixXd PHt = P_ * Ht;
    const MatrixXd S = H_ * PHt + R_;
    const MatrixXd Si = S.inverse();
    const MatrixXd K = PHt * Si;

    // new estimate
    x_ = x_ + (K * y);
    const long x_size = x_.size();
    const MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
  }

  void UpdateEKF(const VectorXd &z) {
    // 현재 H_는 선형 라이다용으로 가정되어 있어 EKF도 동일하게 동작
    const VectorXd z_pred = H_ * x_;
    const VectorXd y = z - z_pred;

    const MatrixXd Ht = H_.transpose();
    const MatrixXd PHt = P_ * Ht;
    const MatrixXd S = H_ * PHt + R_;
    const MatrixXd Si = S.inverse();
    const MatrixXd K = PHt * Si;

    x_ = x_ + (K * y);
    const long x_size = x_.size();
    const MatrixXd I = MatrixXd::Identity(x_size, x_size);
    P_ = (I - K * H_) * P_;
  }

  void check_status() {
    int count = 0;
    std::deque<int> count_deque = fixed_queue.getData();
    const int size = static_cast<int>(count_deque.size());
    const int max_size = fixed_queue.getMaxSize();

    if (size < max_size - 2) {
      std::cout << " Checking..." << std::endl;
      return;
    }

    for (auto e = count_deque.begin(); e != count_deque.end(); ++e) {
      count += *e;
    }

    if (count >= 2) {
      is_dynamic = 1;  // dynamic
    } else {
      is_dynamic = 2;  // static
    }
  }
};

#endif /* KALMAN_FILTER_H_ */

// --------------------------------- FusionEKF ----------------------------------
#ifndef FUSION_EKF_H_
#define FUSION_EKF_H_

using std::vector;

#define PI_ 3.141592

class FusionEKF {
public:
  FusionEKF() {
    is_initialized_ = false;
    previous_timestamp_ = 0;

    // measurement covariance (laser)
    R_laser_ = MatrixXd(2, 2);
    R_laser_ << 0.0225, 0,
                0,      0.0225;

    // measurement matrix (laser)
    H_laser_ = MatrixXd(2, 4);
    H_laser_ << 1, 0, 0, 0,
                0, 1, 0, 0;

    // state covariance
    ekf_.P_ = MatrixXd(4, 4);
    ekf_.P_ << 1, 0, 0, 0,
               0, 1, 0, 0,
               0, 0, 10, 0,
               0, 0, 0, 10;

    // transition
    ekf_.F_ = MatrixXd(4, 4);
    ekf_.F_ << 1, 0, 1, 0,
               0, 1, 0, 1,
               0, 0, 1, 0,
               0, 0, 0, 1;

    // process noise (const accel model)
    noise_ax = 9.0f;
    noise_ay = 9.0f;
  }

  virtual ~FusionEKF() {}

  void ProcessMeasurement(const MeasurementPackage &measurement_pack) {
    // Initialization
    if (!is_initialized_) {
      ekf_.x_ = VectorXd(4);
      // Initialize at measurement (x,y), zero velocity
      ekf_.x_ << measurement_pack.raw_measurements_(0),
                 measurement_pack.raw_measurements_(1),
                 0, 0;

      previous_timestamp_ = measurement_pack.timestamp_;
      is_initialized_ = true;
      return;
    }

    // Prediction step
    const double dt = static_cast<double>(measurement_pack.timestamp_ - previous_timestamp_) / 1e9;
    previous_timestamp_ = measurement_pack.timestamp_;

    const double dt_2 = dt * dt;
    const double dt_3 = dt_2 * dt;
    const double dt_4 = dt_3 * dt;

    // F
    ekf_.F_(0, 2) = dt;
    ekf_.F_(1, 3) = dt;

    // Q
    ekf_.Q_ = MatrixXd(4, 4);
    ekf_.Q_ << dt_4/4*noise_ax,  0,                  dt_3/2*noise_ax,  0,
               0,                dt_4/4*noise_ay,    0,                dt_3/2*noise_ay,
               dt_3/2*noise_ax,  0,                  dt_2*noise_ax,    0,
               0,                dt_3/2*noise_ay,    0,                dt_2*noise_ay;

    ekf_.Predict();

    // Update (Laser)
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  KalmanFilter ekf_;

private:
  bool is_initialized_;
  std::int64_t previous_timestamp_;

  Tools tools;
  Eigen::MatrixXd R_laser_;
  Eigen::MatrixXd H_laser_;

  double noise_ax;
  double noise_ay;
};

#endif /* FUSION_EKF_H_ */
