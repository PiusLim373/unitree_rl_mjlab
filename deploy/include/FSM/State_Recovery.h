#pragma once

#include <chrono>
#include <cmath>

#include "FSM/State_RLBase.h"

// Recovery: an RL policy (loaded like Velocity, from its own policy_dir) that gets
// the robot belly-down into the lying pose, then hands over to Damping
// by itself, either:
//   - done: the robot has held the lying pose for hold_time, or
//   - timeout: it isn't done after `timeout` seconds (training episodes were 8 s;
//     successes take ~2 s). The deterministic policy would only repeat its failing
//     moves, so damping lets the robot settle and L2 + Up retries from a new pose.
//
// Each hand-over is an ordinary FSM check, the same mechanism as the gamepad
// transitions: a condition plus a target state, evaluated every FSM tick (1 kHz);
// the first check that returns true switches state. Thresholds come from
// config.yaml (FSM/<state>/done_check and timeout); done_check matches the
// training metric.
class State_Recovery : public State_RLBase
{
public:
    State_Recovery(int state_mode, std::string state_string)
    : State_RLBase(state_mode, state_string)
    {
        auto cfg = param::config["FSM"][state_string]["done_check"];
        cos_max_tilt_ = std::cos(cfg["max_tilt"].as<float>());
        max_pose_err_ = cfg["max_pose_err"].as<float>();
        max_joint_vel_ = cfg["max_joint_vel"].as<float>();
        hold_time_ = std::chrono::duration<double>(cfg["hold_time"].as<double>());
        lying_pose_ = cfg["lying_pose"].as<std::vector<float>>();  // SDK motor order
        timeout_ = std::chrono::duration<double>(
            param::config["FSM"][state_string]["timeout"].as<double>());

        // Registered after the done check: if both hold on the same tick, done wins.
        this->registered_checks.emplace_back(
            std::make_pair(
                [this]()->bool{
                    if (!lying_still_long_enough()) return false;
                    spdlog::info("Recovery: lying still for {:.1f} s, switching to damping.",
                                 hold_time_.count());
                    return true;
                },
                FSMStringMap.right.at("Damping")
            )
        );
        this->registered_checks.emplace_back(
            std::make_pair(
                [this]()->bool{
                    if (std::chrono::steady_clock::now() - entered_at_ < timeout_) return false;
                    spdlog::warn("Recovery: not done after {:.1f} s, switching to damping "
                                 "(press L2 + Up to retry).", timeout_.count());
                    return true;
                },
                FSMStringMap.right.at("Damping")
            )
        );
    }

    void enter()
    {
        settled_ = false;
        entered_at_ = std::chrono::steady_clock::now();
        State_RLBase::enter();
    }

private:
    // Reads LowState directly: this FSM thread refreshes it every tick (pre_run),
    // while the policy's articulation data belongs to the 50 Hz policy thread.
    bool lying_still_now() const
    {
        const auto & q = lowstate->msg_.imu_state().quaternion();  // w, x, y, z
        // z component of the body's up axis in the world: cos of the tilt.
        float cos_tilt = 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]);
        if (cos_tilt < cos_max_tilt_) return false;
        for (size_t i = 0; i < lying_pose_.size(); ++i)
        {
            const auto & motor = lowstate->msg_.motor_state()[i];
            if (std::fabs(motor.q() - lying_pose_[i]) > max_pose_err_) return false;
            if (std::fabs(motor.dq()) > max_joint_vel_) return false;
        }
        return true;
    }

    // True once lying_still_now() has held continuously for hold_time, so passing
    // through the pose mid-roll doesn't count.
    bool lying_still_long_enough()
    {
        if (!lying_still_now())
        {
            settled_ = false;
            return false;
        }
        auto now = std::chrono::steady_clock::now();
        if (!settled_)
        {
            settled_ = true;
            settled_since_ = now;
        }
        return now - settled_since_ >= hold_time_;
    }

    float cos_max_tilt_;
    float max_pose_err_;
    float max_joint_vel_;
    std::chrono::duration<double> hold_time_;
    std::vector<float> lying_pose_;
    std::chrono::duration<double> timeout_;
    bool settled_ = false;
    std::chrono::steady_clock::time_point settled_since_;
    std::chrono::steady_clock::time_point entered_at_;
};

REGISTER_FSM(State_Recovery)
