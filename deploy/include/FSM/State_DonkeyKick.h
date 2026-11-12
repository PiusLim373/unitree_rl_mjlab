#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>

#include "FSM/State_RLBase.h"

// DonkeyKick: an RL policy (src/tasks/kick, loaded like Velocity from its own
// policy_dir) that tips up onto the front legs (~45 deg nose-down), holds, tucks
// and snaps both rear legs straight back once, comes down and stands still. It is
// started from JointLock or Velocity (standing still) and hands back to that state.
//
// Checks after the gamepad transitions (L2 + B Damping, L2 + Up Recovery),
// evaluated every FSM tick (1 kHz), first true wins:
//   - refused: on entry the robot wasn't level and still (start_check); the policy
//     never starts and nothing is commanded -> back where it came from
//   - fell: pitch, roll or tilt beyond fall_check (the training episode's fall limits)
//     -> the fallback state (Velocity, standing still) while the body is within
//     fallback.max_tilt of level and turning slower than fallback.max_gyro, else Damping
//     (the walking policy catches an upright, slow robot, not one already falling)
//   - done: from done_check.min_time on, level, in the stand pose and still for
//     hold_time (the training STAND_CHECK) -> back where it came from
//   - timeout: not done after `timeout` (the training episode length) -> the fallback
//     state or Damping, as for fell
// A check's target state is fixed, so each hand-back is registered once per origin.
//
// QuickBuck (the fast power kick, no handstand) and SideFlipLeft / SideFlipRight run on
// this same class: each config.yaml entry has type DonkeyKick and its own policy_dir
// (policy_file) and checks. The flips leave roll free (no fall_check.max_roll) and
// limit the tilt only from tilt_after on, as training's flip_fell. FrontFlip and BackFlip
// leave pitch free (no min/max_pitch) and check the left axis instead (max_side_tilt,
// max_side_angle), which holds through vertical, where roll doesn't.
class State_DonkeyKick : public State_RLBase
{
public:
    State_DonkeyKick(int state_mode, std::string state_string)
    : State_RLBase(state_mode, state_string)
    {
        auto cfg = param::config["FSM"][state_string];
        auto start = cfg["start_check"];
        cos_start_tilt_ = std::cos(start["max_tilt"].as<float>());
        start_gyro_ = start["max_gyro"].as<float>();
        start_joint_vel_ = start["max_joint_vel"].as<float>();
        auto fall = cfg["fall_check"];
        check_pitch_ = fall["min_pitch"].IsDefined();
        if (check_pitch_)
        {
            min_pitch_ = fall["min_pitch"].as<float>();
            max_pitch_ = fall["max_pitch"].as<float>();
        }
        check_side_tilt_ = fall["max_side_tilt"].IsDefined();
        if (check_side_tilt_) max_side_tilt_ = fall["max_side_tilt"].as<float>();
        check_side_angle_ = fall["max_side_angle"].IsDefined();
        if (check_side_angle_) cos_side_angle_ = std::cos(fall["max_side_angle"].as<float>());
        check_roll_ = fall["max_roll"].IsDefined();
        if (check_roll_) max_roll_ = fall["max_roll"].as<float>();
        check_tilt_ = fall["max_tilt"].IsDefined();
        if (check_tilt_)
        {
            cos_fall_tilt_ = std::cos(fall["max_tilt"].as<float>());
            tilt_after_ = std::chrono::duration<double>(fall["tilt_after"].as<double>());
        }
        auto done = cfg["done_check"];
        min_time_ = std::chrono::duration<double>(done["min_time"].as<double>());
        cos_done_tilt_ = std::cos(done["max_tilt"].as<float>());
        max_pose_err_ = done["max_pose_err"].as<float>();
        max_joint_vel_ = done["max_joint_vel"].as<float>();
        hold_time_ = std::chrono::duration<double>(done["hold_time"].as<double>());
        stand_pose_ = done["stand_pose"].as<std::vector<float>>();  // SDK motor order
        timeout_ = std::chrono::duration<double>(cfg["timeout"].as<double>());

        if (auto fb = cfg["fallback"])
        {
            const auto name = fb["state"].as<std::string>();
            if (FSMStringMap.right.count(name))
            {
                fallback_ = FSMStringMap.right.at(name);
                cos_fallback_tilt_ = std::cos(fb["max_tilt"].as<float>());
                fallback_gyro_ = fb["max_gyro"].as<float>();
            }
            else
            {
                spdlog::warn("{}: fallback state '{}' not enabled, falls go to Damping.", state_string, name);
            }
        }

        const int damping = FSMStringMap.right.at("Damping");
        std::vector<int> origins;
        for (const char* name : {"JointLock", "Velocity"})
        {
            if (FSMStringMap.right.count(name)) origins.push_back(FSMStringMap.right.at(name));
        }

        for (int origin : origins)
        {
            this->registered_checks.emplace_back(
                [this, origin]()->bool{ return refused_ && came_from_ == origin; }, origin);
        }
        // fell / timeout: the fallback first (only while it can still balance), then Damping.
        if (fallback_ >= 0)
        {
            this->registered_checks.emplace_back(
                [this]()->bool{
                    if (refused_ || !fell_now() || !can_fall_back()) return false;
                    spdlog::warn("{}: tipped too far ({:.0f} deg from level, still catchable), falling back to {}.",
                                 this->getStateString(), tilt_deg(), FSMStringMap.left.at(fallback_));
                    return true;
                },
                fallback_);
        }
        this->registered_checks.emplace_back(
            [this]()->bool{
                if (refused_ || !fell_now()) return false;
                spdlog::warn("{}: tipped too far ({:.0f} deg from level), switching to damping "
                             "(press L2 + Up to recover).", this->getStateString(), tilt_deg());
                return true;
            },
            damping);
        for (int origin : origins)
        {
            this->registered_checks.emplace_back(
                [this, origin]()->bool{
                    if (refused_ || came_from_ != origin || !standing_long_enough()) return false;
                    spdlog::info("{}: done, standing still; back to {}.",
                                 this->getStateString(), FSMStringMap.left.at(origin));
                    return true;
                },
                origin);
        }
        if (fallback_ >= 0)
        {
            this->registered_checks.emplace_back(
                [this]()->bool{
                    if (refused_ || !timed_out() || !can_fall_back()) return false;
                    spdlog::warn("{}: not standing still after {:.1f} s, falling back to {}.",
                                 this->getStateString(), timeout_.count(), FSMStringMap.left.at(fallback_));
                    return true;
                },
                fallback_);
        }
        this->registered_checks.emplace_back(
            [this]()->bool{
                if (refused_ || !timed_out()) return false;
                spdlog::warn("{}: not standing still after {:.1f} s, switching to damping.",
                             this->getStateString(), timeout_.count());
                return true;
            },
            damping);
    }

    void enter()
    {
        came_from_ = BaseState::previous_state;
        settled_ = false;
        entered_at_ = std::chrono::steady_clock::now();
        // The left axis's start direction, made level (start_check keeps it near level).
        const auto side = side_axis();
        const float h = std::max(std::hypot(side[0], side[1]), 1e-6f);
        side0_ = {side[0] / h, side[1] / h};
        refused_ = !level_and_still();
        if (refused_)
        {
            spdlog::warn("{}: refused, the robot isn't standing level and still; back to {}.",
                         getStateString(),
                         FSMStringMap.left.count(came_from_) ? FSMStringMap.left.at(came_from_) : "?");
            return;
        }
        spdlog::info("{}: go.", getStateString());
        State_RLBase::enter();
    }

    // Refused: command nothing, so the previous state's last command holds for the
    // one tick before the hand-back.
    void run()
    {
        if (!refused_) State_RLBase::run();
    }

private:
    // These read LowState directly: this FSM thread refreshes it every tick
    // (pre_run), while the policy's articulation data belongs to the 50 Hz thread.

    // Pitch (+ = nose down) and roll from the IMU quaternion (w, x, y, z), the same
    // angles as training's pitch_roll (from projected gravity). The pitch is the
    // forward axis's angle from level, valid at any roll (upside down too).
    void pitch_roll(float & pitch, float & roll) const
    {
        const auto & q = lowstate->msg_.imu_state().quaternion();
        pitch = std::asin(std::clamp(2.0f * (q[0] * q[2] - q[1] * q[3]), -1.0f, 1.0f));
        roll = std::atan2(2.0f * (q[2] * q[3] + q[0] * q[1]), 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]));
    }

    // The body's left axis in the world from the IMU quaternion (w, x, y, z): the front
    // flip's axis, as training's _body_axis(axis=1).
    std::array<float, 3> side_axis() const
    {
        const auto & q = lowstate->msg_.imu_state().quaternion();
        return {2.0f * (q[1] * q[2] - q[0] * q[3]), 1.0f - 2.0f * (q[1] * q[1] + q[3] * q[3]),
                2.0f * (q[2] * q[3] + q[0] * q[1])};
    }

    float cos_tilt() const
    {
        const auto & q = lowstate->msg_.imu_state().quaternion();
        return 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]);
    }

    float tilt_deg() const
    {
        return std::acos(std::clamp(cos_tilt(), -1.0f, 1.0f)) * 180.0f / static_cast<float>(M_PI);
    }

    float max_joint_speed() const
    {
        float v = 0.0f;
        for (size_t i = 0; i < stand_pose_.size(); ++i)
            v = std::max(v, std::fabs(lowstate->msg_.motor_state()[i].dq()));
        return v;
    }

    bool level_and_still() const
    {
        if (cos_tilt() < cos_start_tilt_) return false;
        const auto & w = lowstate->msg_.imu_state().gyroscope();
        if (std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) > start_gyro_) return false;
        return max_joint_speed() <= start_joint_vel_;
    }

    bool fell_now() const
    {
        float pitch, roll;
        pitch_roll(pitch, roll);
        if (check_pitch_ && (pitch < min_pitch_ || pitch > max_pitch_)) return true;
        if (check_roll_ && std::fabs(roll) > max_roll_) return true;
        if (check_side_tilt_ || check_side_angle_)
        {
            const auto side = side_axis();
            // Tipped from level (training's max_axis_tilt), or tipped or turned from its
            // start direction (max_axis_angle).
            if (check_side_tilt_ && std::fabs(std::asin(std::clamp(side[2], -1.0f, 1.0f))) > max_side_tilt_)
                return true;
            if (check_side_angle_ && side[0] * side0_[0] + side[1] * side0_[1] < cos_side_angle_) return true;
        }
        return check_tilt_ && std::chrono::steady_clock::now() - entered_at_ >= tilt_after_
            && cos_tilt() < cos_fall_tilt_;
    }

    bool timed_out() const
    {
        return std::chrono::steady_clock::now() - entered_at_ >= timeout_;
    }

    // The fallback (walking policy) only takes over a robot it can still balance.
    bool can_fall_back() const
    {
        if (fallback_ < 0 || cos_tilt() < cos_fallback_tilt_) return false;
        const auto & w = lowstate->msg_.imu_state().gyroscope();
        return std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) <= fallback_gyro_;
    }

    bool standing_now() const
    {
        if (cos_tilt() < cos_done_tilt_) return false;
        for (size_t i = 0; i < stand_pose_.size(); ++i)
        {
            if (std::fabs(lowstate->msg_.motor_state()[i].q() - stand_pose_[i]) > max_pose_err_) return false;
        }
        return max_joint_speed() < max_joint_vel_;
    }

    // True once standing_now() has held for hold_time, counting from min_time (the
    // kick itself passes through level poses).
    bool standing_long_enough()
    {
        auto now = std::chrono::steady_clock::now();
        if (now - entered_at_ < min_time_ || !standing_now())
        {
            settled_ = false;
            return false;
        }
        if (!settled_)
        {
            settled_ = true;
            settled_since_ = now;
        }
        return now - settled_since_ >= hold_time_;
    }

    float cos_start_tilt_, start_gyro_, start_joint_vel_;
    float min_pitch_ = 0.0f, max_pitch_ = 0.0f, max_roll_ = 0.0f, cos_fall_tilt_ = -1.0f;
    float max_side_tilt_ = 0.0f, cos_side_angle_ = -1.0f;
    bool check_pitch_ = false, check_roll_ = false, check_tilt_ = false;
    bool check_side_tilt_ = false, check_side_angle_ = false;
    std::array<float, 2> side0_{0.0f, 1.0f};
    std::chrono::duration<double> min_time_, hold_time_, timeout_, tilt_after_{0.0};
    float cos_done_tilt_, max_pose_err_, max_joint_vel_;
    std::vector<float> stand_pose_;
    int fallback_ = -1;
    float cos_fallback_tilt_ = 1.0f, fallback_gyro_ = 0.0f;

    int came_from_ = 0;
    bool refused_ = false;
    bool settled_ = false;
    std::chrono::steady_clock::time_point entered_at_, settled_since_;
};

REGISTER_FSM(State_DonkeyKick)
