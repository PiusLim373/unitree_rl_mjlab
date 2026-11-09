#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include <eigen3/Eigen/Dense>

// UprightWalk's angles and the paw's gravity torque. Only needs Eigen, so it can be
// checked on its own against training (mdp/frames.py) and MuJoCo (qfrc_bias).
namespace upright
{

inline Eigen::Quaternionf to_quat(const std::array<float, 4> & wxyz)
{
    return Eigen::Quaternionf(wxyz[0], wxyz[1], wxyz[2], wxyz[3]);
}

// The nose's angle up from the heading direction: 0 on four legs, pi/2 upright, past
// pi/2 leaning back (training's nose_angle). The heading comes from the left axis, which
// stays level in both postures: forward = left x up.
inline float nose_angle(const Eigen::Quaternionf & q)
{
    const Eigen::Vector3f nose = q * Eigen::Vector3f::UnitX();
    Eigen::Vector3f left = q * Eigen::Vector3f::UnitY();
    left.z() = 0.0f;
    left /= std::max(left.norm(), 1e-6f);
    const Eigen::Vector3f forward = left.cross(Eigen::Vector3f::UnitZ());
    return std::atan2(nose.z(), nose.dot(forward));
}

// How far the left axis tips from level (training's roll_angle).
inline float roll_angle(const Eigen::Quaternionf & q)
{
    return std::asin(std::clamp((q * Eigen::Vector3f::UnitY()).z(), -1.0f, 1.0f));
}

// Torque (hip, thigh, calf) that holds a front leg against gravity, for the soft paw
// (training adds qfrc_bias). side: +1 left, -1 right; q: the leg's joints; quat: the body.
// Masses and centres of mass are go2.xml's front leg.
inline Eigen::Vector3f paw_gravity_torque(float side, const Eigen::Vector3f & q, const Eigen::Quaternionf & quat)
{
    constexpr float kG = 9.81f;
    constexpr float kHipMass = 0.678f, kThighMass = 1.152f, kCalfMass = 0.2414f;
    const Eigen::Vector3f g = quat.conjugate() * Eigen::Vector3f(0.0f, 0.0f, -kG);

    const Eigen::Matrix3f r1 = Eigen::AngleAxisf(q[0], Eigen::Vector3f::UnitX()).toRotationMatrix();
    const Eigen::Matrix3f r2 = r1 * Eigen::AngleAxisf(q[1], Eigen::Vector3f::UnitY()).toRotationMatrix();
    const Eigen::Matrix3f r3 = r2 * Eigen::AngleAxisf(q[2], Eigen::Vector3f::UnitY()).toRotationMatrix();

    // From the hip joint: the thigh and calf joints, then each link's centre of mass.
    const Eigen::Vector3f p2 = r1 * Eigen::Vector3f(0.0f, side * 0.0955f, 0.0f);
    const Eigen::Vector3f p3 = p2 + r2 * Eigen::Vector3f(0.0f, 0.0f, -0.213f);
    const Eigen::Vector3f c1 = r1 * Eigen::Vector3f(-0.0054f, side * 0.00194f, -0.0001f);
    const Eigen::Vector3f c2 = p2 + r2 * Eigen::Vector3f(-0.00374f, side * -0.0223f, -0.0327f);
    const Eigen::Vector3f c3 = p3 + r3 * Eigen::Vector3f(0.0063f, side * -0.00062f, -0.14142f);

    const Eigen::Vector3f axis1 = Eigen::Vector3f::UnitX();
    const Eigen::Vector3f axis2 = r1 * Eigen::Vector3f::UnitY();
    const Eigen::Vector3f axis3 = r2 * Eigen::Vector3f::UnitY();

    // Gravity's torque about each joint from the links after it; the motor gives the opposite.
    const Eigen::Vector3f m3 = (c3 - p3).cross(kCalfMass * g);
    const Eigen::Vector3f m2 = (c2 - p2).cross(kThighMass * g) + (c3 - p2).cross(kCalfMass * g);
    const Eigen::Vector3f m1 = c1.cross(kHipMass * g) + c2.cross(kThighMass * g) + c3.cross(kCalfMass * g);
    return Eigen::Vector3f(-axis1.dot(m1), -axis2.dot(m2), -axis3.dot(m3));
}

}  // namespace upright
