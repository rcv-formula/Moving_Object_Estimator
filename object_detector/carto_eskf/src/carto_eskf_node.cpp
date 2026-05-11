#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>  // std::min/std::max
#include <cmath>      // sin, cos, acos, M_PI
#include <limits>

using sensor_msgs::msg::Imu;
using nav_msgs::msg::Odometry;

struct State {
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};
  Eigen::Vector3d v{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d bg{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ba{Eigen::Vector3d::Zero()};
  double last_ts{-1.0};  // [s]
};

class CartoESKFNode : public rclcpp::Node {
public:
  CartoESKFNode() : Node("carto_eskf_node") {
    // --- Params (fallback 및 게이팅/재초기화) ---
    sigma_p_ = this->declare_parameter("meas_sigma_pos", 0.03);                 // [m]
    sigma_th_= this->declare_parameter("meas_sigma_ang", 0.5*M_PI/180.0);       // [rad]
    gate_prob_= this->declare_parameter("gate_prob", 0.9973);                   // ≈3σ (표시용)
    adapt_scale_= this->declare_parameter("adapt_scale", 10.0);
    jump_reject_norm_= this->declare_parameter("jump_reject_pos_norm", 1.5);    // [m]
    reinit_on_jump_ = this->declare_parameter("reinit_on_jump", true);          // 점프시 재초기화
    reinit_pos_thresh_ = this->declare_parameter("reinit_pos_thresh", 3.0);     // [m]

    // IMU 잡음 (AirIMU 보정 이후 잔여)
    na_ = this->declare_parameter("noise_acc", 0.6);
    ng_ = this->declare_parameter("noise_gyr", 0.02);
    nba_= this->declare_parameter("noise_ba", 0.01);
    nbg_= this->declare_parameter("noise_bg", 0.001);

    P_.setIdentity(15,15); P_ *= 1e-2;

    imu_sub_ = this->create_subscription<Imu>(
      "/airimu_imu_data", rclcpp::SensorDataQoS(),
      std::bind(&CartoESKFNode::imuCb, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<Odometry>(
      "/odom", 10, std::bind(&CartoESKFNode::odomCb, this, std::placeholders::_1));

    odom_pub_ = this->create_publisher<Odometry>("/processed_odom", 10);
  }

private:
  // ---- 소도구 ----
  static Eigen::Matrix3d Skew(const Eigen::Vector3d& v){
    Eigen::Matrix3d S; S << 0,-v.z(),v.y(), v.z(),0,-v.x(), -v.y(),v.x(),0; return S;
  }
  static Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& w){
    double th=w.norm(); if(th<1e-12) return Eigen::Matrix3d::Identity()+Skew(w);
    Eigen::Vector3d a=w/th; Eigen::Matrix3d A=Skew(a);
    return Eigen::Matrix3d::Identity()+sin(th)*A+(1-cos(th))*A*A;
  }
  static Eigen::Vector3d LogSO3(const Eigen::Matrix3d& R){
    double c=(R.trace()-1)*0.5; c=std::min(1.0,std::max(-1.0,c));
    double th=acos(c);
    if(th<1e-12) return Eigen::Vector3d::Zero();
    Eigen::Vector3d w; w << R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1);
    return 0.5*th/sin(th)*w;
  }

  // ---- /odom으로 초기화 또는 재초기화 ----
  void initializeFromOdom(const Odometry& m, const char* reason="init"){
    state_.p = Eigen::Vector3d(
      m.pose.pose.position.x,
      m.pose.pose.position.y,
      m.pose.pose.position.z);
    state_.q = Eigen::Quaterniond(
      m.pose.pose.orientation.w,
      m.pose.pose.orientation.x,
      m.pose.pose.orientation.y,
      m.pose.pose.orientation.z);
    state_.q.normalize();
    state_.v.setZero();
    state_.bg.setZero();
    state_.ba.setZero();

    // 공분산 리셋(초기 신뢰 높게 시작)
    P_.setIdentity(15,15);
    P_.block<3,3>(0,0)   *= 1e-4;  // p
    P_.block<3,3>(3,3)   *= 1e-2;  // v
    P_.block<3,3>(6,6)   *= 1e-4;  // theta
    P_.block<3,3>(9,9)   *= 1e-3;  // bg
    P_.block<3,3>(12,12) *= 1e-2;  // ba

    // 시간 기준도 odom 시각으로 맞춤 (선택)
    state_.last_ts = rclcpp::Time(m.header.stamp).seconds();

    initialized_ = true;

    publishOdom(m.header.stamp);
    RCLCPP_INFO(this->get_logger(),
      "[%s] from /odom: p=(%.3f,%.3f,%.3f)",
      reason, state_.p.x(), state_.p.y(), state_.p.z());
  }

  // ---- IMU 예측 ----
  void imuCb(const Imu::SharedPtr m){
    const double t = rclcpp::Time(m->header.stamp).seconds();
    if(state_.last_ts<0){ state_.last_ts=t; return; } // 초기 타임스탬프만 설정
    const double dt = std::max(1e-4, t - state_.last_ts);
    state_.last_ts = t;

    Eigen::Vector3d wm(m->angular_velocity.x, m->angular_velocity.y, m->angular_velocity.z);
    Eigen::Vector3d am(m->linear_acceleration.x, m->linear_acceleration.y, m->linear_acceleration.z);
    Eigen::Vector3d w = wm - state_.bg;
    Eigen::Vector3d a = am - state_.ba;

    Eigen::Matrix3d R = state_.q.toRotationMatrix();
    state_.p += state_.v*dt + 0.5*(R*a + g_)*dt*dt;
    state_.v += (R*a + g_)*dt;

    Eigen::Matrix3d dR = ExpSO3(w*dt);
    state_.q = Eigen::Quaterniond(dR) * state_.q;
    state_.q.normalize();

    // 공분산 전파(1차 선형화)
    Eigen::Matrix<double,15,15> F = Eigen::Matrix<double,15,15>::Zero();
    F.block<3,3>(0,3) = Eigen::Matrix3d::Identity();
    F.block<3,3>(0,6) = -R*Skew(a);
    F.block<3,3>(0,12)= -R;
    F.block<3,3>(3,6) = -R*Skew(a);
    F.block<3,3>(3,12)= -R;
    F.block<3,3>(6,6) = -Skew(w);
    F.block<3,3>(6,9) = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double,15,12> G = Eigen::Matrix<double,15,12>::Zero();
    G.block<3,3>(6,0) = Eigen::Matrix3d::Identity(); // gyro noise
    G.block<3,3>(3,3) = R;                           // acc noise (dv)
    G.block<3,3>(0,3) = 0.5*R*dt;                    // acc noise (dp)
    G.block<3,3>(9,6) = Eigen::Matrix3d::Identity(); // bg RW
    G.block<3,3>(12,9)= Eigen::Matrix3d::Identity(); // ba RW

    Eigen::Matrix<double,12,12> Q = Eigen::Matrix<double,12,12>::Zero();
    Q.block<3,3>(0,0) = (ng_*ng_)*Eigen::Matrix3d::Identity();
    Q.block<3,3>(3,3) = (na_*na_)*Eigen::Matrix3d::Identity();
    Q.block<3,3>(6,6) = (nbg_*nbg_)*Eigen::Matrix3d::Identity();
    Q.block<3,3>(9,9) = (nba_*nba_)*Eigen::Matrix3d::Identity();

    const Eigen::Matrix<double,15,15> I = Eigen::Matrix<double,15,15>::Identity();
    const Eigen::Matrix<double,15,15> Fd = I + F*dt;
    const Eigen::Matrix<double,15,15> Qd = G*Q*G.transpose()*dt;

    P_ = Fd*P_*Fd.transpose() + Qd;

    publishOdom(m->header.stamp);
  }

  // ---- Cartographer /odom.pose 업데이트 ----
  void odomCb(const Odometry::SharedPtr m){
    // 초기화가 아직 안 됐다면 첫 /odom으로 초기화
    if(!initialized_){
      initializeFromOdom(*m, "init");
      return; // 초기화 프레임은 업데이트 생략
    }

    Eigen::Vector3d pz(m->pose.pose.position.x, m->pose.pose.position.y, m->pose.pose.position.z);
    const auto& qmsg = m->pose.pose.orientation;
    Eigen::Quaterniond qz(qmsg.w, qmsg.x, qmsg.y, qmsg.z);

    // 메시지 공분산 사용 시도
    Eigen::Matrix<double,6,6> Rm = Eigen::Matrix<double,6,6>::Zero();
    const auto& C = m->pose.covariance;
    double sum=0; for(int i=0;i<36;i++) sum+=std::abs(C[i]);
    if(sum>0 && std::isfinite(sum)){
      for(int r=0;r<6;r++) for(int c=0;c<6;c++) Rm(r,c)=C[r*6+c];
      for(int i=0;i<3;i++){
        Rm(i,i)=std::max(Rm(i,i),1e-4);
        Rm(i+3,i+3)=std::max(Rm(i+3,i+3),(0.2*M_PI/180.0)*(0.2*M_PI/180.0));
      }
    }else{
      for(int i=0;i<3;i++){ Rm(i,i)=sigma_p_*sigma_p_; Rm(i+3,i+3)=sigma_th_*sigma_th_; }
    }

    // 혁신
    Eigen::Vector3d rp = pz - state_.p;

    // (옵션) 큰 점프 시 재초기화 트리거
    if(reinit_on_jump_ && rp.norm() > reinit_pos_thresh_){
      RCLCPP_WARN(this->get_logger(),
        "Reinit triggered: |dp|=%.2f m > %.2f m", rp.norm(), reinit_pos_thresh_);
      initializeFromOdom(*m, "reinit");
      return;
    }

    // (업데이트 전) 하드 리젝트(짧은 튐 억제)
    if(rp.norm() > jump_reject_norm_){
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
        "Carto jump: |dp|=%.2f m → reject", rp.norm());
      return;
    }

    Eigen::Vector3d rth = LogSO3(state_.q.toRotationMatrix().transpose() * qz.toRotationMatrix());
    Eigen::Matrix<double,6,1> r; r << rp, rth;

    // H, S, K
    Eigen::Matrix<double,6,15> H = Eigen::Matrix<double,6,15>::Zero();
    H.block<3,3>(0,0) = Eigen::Matrix3d::Identity(); // p
    H.block<3,3>(3,6) = Eigen::Matrix3d::Identity(); // theta

    Eigen::Matrix<double,6,6> S = H*P_*H.transpose() + Rm;
    Eigen::Matrix<double,15,6> K = P_*H.transpose()*S.inverse();

    // 마할라노비스 게이팅 (간이 임계값: dof=6 → ~16.81 @ 0.9973)
    const double gamma = r.transpose()*S.inverse()*r;
    const double gate = 16.81;
    if(gamma > gate){
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1500,
        "Gate: gamma=%.2f > %.2f → down-weight", gamma, gate);
      Eigen::Matrix<double,6,6> S2 = H*P_*H.transpose() + (adapt_scale_*Rm);
      K = P_*H.transpose()*S2.inverse();
    }

    // 상태/공분산 업데이트
    Eigen::Matrix<double,15,1> dx = K*r;
    state_.p  += dx.block<3,1>(0,0);
    state_.v  += dx.block<3,1>(3,0);
    state_.q   = Eigen::Quaterniond(ExpSO3(dx.block<3,1>(6,0))) * state_.q;
    state_.q.normalize();
    state_.bg += dx.block<3,1>(9,0);
    state_.ba += dx.block<3,1>(12,0);

    const Eigen::Matrix<double,15,15> I = Eigen::Matrix<double,15,15>::Identity();
    P_ = (I - K*H)*P_*(I - K*H).transpose() + K*Rm*K.transpose();

    publishOdom(m->header.stamp);
  }

  void publishOdom(const rclcpp::Time& stamp){
    Odometry o;
    o.header.stamp = stamp;
    o.header.frame_id = "map";
    o.child_frame_id = "base_link";
    o.pose.pose.position.x = state_.p.x();
    o.pose.pose.position.y = state_.p.y();
    o.pose.pose.position.z = state_.p.z();
    o.pose.pose.orientation.w = state_.q.w();
    o.pose.pose.orientation.x = state_.q.x();
    o.pose.pose.orientation.y = state_.q.y();
    o.pose.pose.orientation.z = state_.q.z();
    for(int i=0;i<6;i++) o.pose.covariance[i*6+i] = (i<3 ? P_(i,i) : P_(i+3,i+3));
    odom_pub_->publish(o);
  }

  // ---- 멤버 ----
  State state_;
  Eigen::Matrix<double,15,15> P_;
  Eigen::Vector3d g_{0,0,-9.80665};

  bool initialized_{false};
  bool reinit_on_jump_{true};
  double reinit_pos_thresh_{3.0};

  double sigma_p_, sigma_th_, gate_prob_, adapt_scale_, jump_reject_norm_;
  double na_, ng_, nba_, nbg_;

  rclcpp::Subscription<Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<Odometry>::SharedPtr odom_pub_;
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CartoESKFNode>());
  rclcpp::shutdown();
  return 0;
}
