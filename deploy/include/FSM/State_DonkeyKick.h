#pragma once

#include <algorithm>
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
//   - fell: pitch or roll beyond fall_check (the training episode's fall limits) -> Damping
//   - done: from done_check.min_time on, level, in the stand pose and still for
//     hold_time (the training STAND_CHECK) -> back where it came from
//   - timeout: not done after `timeout` (the training episode length) -> Damping
// A check's target state is fixed, so each hand-back is registered once per origin.
//
// QuickBuck (the fast power kick, no handstand) runs on this same class: its
// config.yaml entry has type DonkeyKick and its own policy_dir and checks.
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
        min_pitch_ = fall["min_pitch"].as<float>();
        max_pitch_ = fall["max_pitch"].as<float>();
        max_roll_ = fall["max_roll"].as<float>();
        auto done = cfg["done_check"];
        min_time_ = std::chrono::duration<double>(done["min_time"].as<double>());
        cos_done_tilt_ = std::cos(done["max_tilt"].as<float>());
        max_pose_err_ = done["max_pose_err"].as<float>();
        max_joint_vel_ = done["max_joint_vel"].as<float>();
        hold_time_ = std::chrono::duration<double>(done["hold_time"].as<double>());
        stand_pose_ = done["stand_pose"].as<std::vector<float>>();  // SDK motor order
        timeout_ = std::chrono::duration<double>(cfg["timeout"].as<double>());

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
        this->registered_checks.emplace_back(
            [this]()->bool{
                if (refused_ || !fell_now()) return false;
                spdlog::warn("{}: tipped too far, switching to damping "
                             "(press L2 + Up to recover).", this->getStateString());
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
        this->registered_checks.emplace_back(
            [this]()->bool{
                if (refused_ || std::chrono::steady_clock::now() - entered_at_ < timeout_) return false;
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
    // angles as training's pitch_roll (from projected gravity).
    void pitch_roll(float & pitch, float & roll) const
    {
        const auto & q = lowstate->msg_.imu_state().quaternion();
        pitch = std::asin(std::clamp(2.0f * (q[0] * q[2] - q[1] * q[3]), -1.0f, 1.0f));
        roll = std::atan2(2.0f * (q[2] * q[3] + q[0] * q[1]), 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]));
    }

    float cos_tilt() const
    {
        const auto & q = lowstate->msg_.imu_state().quaternion();
        return 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]);
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
        return pitch < min_pitch_ || pitch > max_pitch_ || std::fabs(roll) > max_roll_;
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
    float min_pitch_, max_pitch_, max_roll_;
    std::chrono::duration<double> min_time_, hold_time_, timeout_;
    float cos_done_tilt_, max_pose_err_, max_joint_vel_;
    std::vector<float> stand_pose_;

    int came_from_ = 0;
    bool refused_ = false;
    bool settled_ = false;
    std::chrono::steady_clock::time_point entered_at_, settled_since_;
};

REGISTER_FSM(State_DonkeyKick)
