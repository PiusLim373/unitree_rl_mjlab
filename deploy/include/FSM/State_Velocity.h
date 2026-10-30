#pragma once

#include <algorithm>
#include <cmath>

#include "FSM/State_RLBase.h"

// Velocity: the walking policy (RLBase) plus a guard on the way back to JointLock
// (Start or L2 + A on the Go2). The button press only counts while the robot is
// stopped (FSM/<state>/stop_check): sticks centred, no D-pad held, every joint
// nearly still. Otherwise the press is ignored with a warning and the policy keeps
// walking, so a bumped Start can't freeze the legs mid-stride.
class State_Velocity : public State_RLBase
{
public:
    State_Velocity(int state_mode, std::string state_string)
    : State_RLBase(state_mode, state_string)
    {
        auto stop = param::config["FSM"][state_string]["stop_check"];
        max_stick_ = stop["max_stick"].as<float>();
        max_joint_vel_ = stop["max_joint_vel"].as<float>();

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

    float max_stick_, max_joint_vel_;
};

REGISTER_FSM(State_Velocity)
