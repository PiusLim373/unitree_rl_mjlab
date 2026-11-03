#pragma once

#include <algorithm>
#include <cmath>

#include "FSM/State_RLBase.h"
#include "isaaclab/envs/mdp/observations/observations.h"

// Velocity: the walking policy (RLBase) plus a guard on the way back to JointLock
// (Start or L2 + A on the Go2). The button press only counts while the robot is
// stopped (FSM/<state>/stop_check): sticks centred, no D-pad held, every joint
// nearly still. Otherwise the press is ignored with a warning and the policy keeps
// walking, so a bumped Start can't freeze the legs mid-stride.
//
// Optional:
//   fall_check.max_tilt: body further than this from level -> Damping (training's
//     fell_over limit; the policy never saw beyond it)
//   zero_commands_from: entered from one of these states (the skills, which hand over
//     here when done or as their fallback), the command reads 0 until the sticks are
//     centred and the D-pad released once, so a held skill combo doesn't drive it
class State_Velocity : public State_RLBase
{
public:
    State_Velocity(int state_mode, std::string state_string)
    : State_RLBase(state_mode, state_string)
    {
        auto cfg = param::config["FSM"][state_string];
        auto stop = cfg["stop_check"];
        max_stick_ = stop["max_stick"].as<float>();
        max_joint_vel_ = stop["max_joint_vel"].as<float>();
        if (cfg["zero_commands_from"])
            zero_commands_from_ = cfg["zero_commands_from"].as<std::vector<std::string>>();

        if (auto fall = cfg["fall_check"])
        {
            cos_fall_tilt_ = std::cos(fall["max_tilt"].as<float>());
            this->registered_checks.emplace_back(
                [this]()->bool{
                    if (cos_tilt() >= cos_fall_tilt_) return false;
                    spdlog::warn("{}: fell over, switching to damping (press L2 + Up to recover).",
                                 this->getStateString());
                    return true;
                },
                FSMStringMap.right.at("Damping"));
        }

        if (!FSMStringMap.right.count("JointLock")) return;
        const int joint_lock = FSMStringMap.right.at("JointLock");
        for (auto & check : this->registered_checks)
        {
            if (check.second != joint_lock) continue;
            auto pressed = check.first;
            check.first = [this, pressed]()->bool{
                if (!pressed()) return false;
                if (stopped()) return true;
                spdlog::warn("{}: stop first (centre the sticks), then Start / L2 + A.",
                             this->getStateString());
                return false;
            };
        }
    }

    void enter()
    {
        const auto & from = FSMStringMap.left.count(BaseState::previous_state)
            ? FSMStringMap.left.at(BaseState::previous_state) : std::string();
        const bool hold = std::count(zero_commands_from_.begin(), zero_commands_from_.end(), from) > 0;
        isaaclab::mdp::hold_velocity_commands = hold;
        if (hold) spdlog::info("{}: from {}, standing until the sticks are centred and the D-pad released.",
                               getStateString(), from);
        State_RLBase::enter();
    }

private:
    // Reads LowState directly: this FSM thread refreshes it every tick (pre_run).
    bool stopped() const
    {
        auto & js = lowstate->joystick;
        if (std::fabs(js.lx()) > max_stick_ || std::fabs(js.ly()) > max_stick_
            || std::fabs(js.rx()) > max_stick_) return false;
        if (js.up.pressed || js.down.pressed || js.left.pressed || js.right.pressed) return false;
        for (int i = 0; i < 12; ++i)
        {
            if (std::fabs(lowstate->msg_.motor_state()[i].dq()) > max_joint_vel_) return false;
        }
        return true;
    }

    // z component of the body's up axis in the world: cos of the tilt.
    float cos_tilt() const
    {
        const auto & q = lowstate->msg_.imu_state().quaternion();  // w, x, y, z
        return 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]);
    }

    float max_stick_, max_joint_vel_;
    float cos_fall_tilt_ = -1.0f;
    std::vector<std::string> zero_commands_from_;
};

REGISTER_FSM(State_Velocity)
