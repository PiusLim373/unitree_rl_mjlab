// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <algorithm>
#include <cmath>

#include "FSMState.h"
#include "LinearInterpolator.h"

// Moves the joints through keyframes (FSM/<state>/ts, qs; qs[0] is replaced by the
// current command on entry) with fixed gains (kp, kd), then holds the last one.
// Each state reads its own block, so one class serves several poses: JointLock, and
// on the Go2 LieDown (type JointLock, lying pose).
//
// With an optional done_check block (Go2 LieDown) it hands over to Damping
// by itself instead of holding: once the keyframes are done and every joint has held
// the final one for hold_time, or after timeout either way. Same mechanism as
// Recovery's done check; registered after the gamepad transitions, so a press on the
// same tick wins.
class State_JointLock : public FSMState
{
public:
    State_JointLock(int state, std::string state_string = "JointLock")
    : FSMState(state, state_string)
    {
        auto cfg = param::config["FSM"][state_string];
        kp_ = cfg["kp"].as<std::vector<float>>();
        kd_ = cfg["kd"].as<std::vector<float>>();
        ts_full_ = cfg["ts"].as<std::vector<float>>();
        qs_full_ = cfg["qs"].as<std::vector<std::vector<float>>>();
        assert(ts_full_.size() == qs_full_.size());
        if(cfg["straight_from"]) {
            straight_from_ = cfg["straight_from"].as<std::vector<std::string>>();
        }
        if(cfg["done_check"]) {
            auto done = cfg["done_check"];
            max_pose_err_ = done["max_pose_err"].as<float>();
            max_joint_vel_ = done["max_joint_vel"].as<float>();
            hold_time_ = done["hold_time"].as<double>();
            timeout_ = done["timeout"].as<double>();

            this->registered_checks.emplace_back(
                std::make_pair(
                    [this]()->bool{
                        if(!held_final_pose()) return false;
                        spdlog::info("{}: holding the final pose, switching to damping.", getStateString());
                        return true;
                    },
                    FSMStringMap.right.at("Damping")
                )
            );
            this->registered_checks.emplace_back(
                std::make_pair(
                    [this]()->bool{
                        if(elapsed() < timeout_) return false;
                        spdlog::warn("{}: final pose not reached after {:.1f} s, switching to damping.",
                                     getStateString(), timeout_);
                        return true;
                    },
                    FSMStringMap.right.at("Damping")
                )
            );
        }
    }

    void enter()
    {
        // set gain
        for(int i(0); i < kp_.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = kp_[i];
            motor.kd() = kd_[i];
            motor.dq() = motor.tau() = 0;
        }


        // set initial position
        std::vector<float> q0;
        for(int i(0); i < kp_.size(); ++i) {
            q0.push_back(lowcmd->msg_.motor_cmd()[i].q());
        }
        ts_ = ts_full_;
        qs_ = qs_full_;
        qs_[0] = q0;

        // Entered from a state listed in straight_from (Go2 JointLock: Velocity and the
        // kicks) with the robot already near the final keyframe: go straight there in
        // 1 s instead of crouching down to the first one and back up. Measured joints,
        // so a policy's command offsets don't hide a standing robot. Every other way in
        // (and every state without straight_from) runs the full keyframes.
        bool listed = false;
        if(FSMStringMap.left.count(previous_state)) {
            listed = std::count(straight_from_.begin(), straight_from_.end(),
                                FSMStringMap.left.at(previous_state)) > 0;
        }
        float err = 0.0f;
        for(int i(0); i < kp_.size(); ++i) {
            err = std::max(err, std::fabs(lowstate->msg_.motor_state()[i].q() - qs_full_.back()[i]));
        }
        if(listed && err < kStandingErr) {
            ts_ = {0.0f, 1.0f};
            qs_ = {q0, qs_full_.back()};
            spdlog::info("{}: already there, going straight to the final pose.", getStateString());
        }
        settled_ = false;
        t0_ = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;
    }

    void run()
    {
        float t = elapsed();
        auto q = linear_interpolate(t, ts_, qs_);

        for(int i(0); i < q.size(); ++i) {
            lowcmd->msg_.motor_cmd()[i].q() = q[i];
        }
    }

private:
    // s since enter()
    double elapsed() const
    {
        return (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3 - t0_;
    }

    // True once the keyframes are done and every joint has stayed within max_pose_err
    // of the final keyframe, slower than max_joint_vel, for hold_time. Measured joints
    // (LowState, refreshed every FSM tick), so a blocked leg doesn't count as there.
    bool held_final_pose()
    {
        const double t = elapsed();
        bool there = t >= ts_.back();
        for(int i(0); there && i < kp_.size(); ++i) {
            const auto & motor = lowstate->msg_.motor_state()[i];
            there = std::fabs(motor.q() - qs_.back()[i]) <= max_pose_err_
                 && std::fabs(motor.dq()) <= max_joint_vel_;
        }
        if(!there) {
            settled_ = false;
            return false;
        }
        if(!settled_) {
            settled_ = true;
            settled_since_ = t;
        }
        return t - settled_since_ >= hold_time_;
    }

    // rad, every joint vs the final keyframe. Lying (JointLock's first keyframe) is
    // ~1.15 rad away at the calves.
    static constexpr float kStandingErr = 0.5f;

    double t0_;
    std::vector<float> kp_, kd_;
    std::vector<std::string> straight_from_;
    std::vector<float> ts_full_, ts_;
    std::vector<std::vector<float>> qs_full_, qs_;

    // done_check (only read when the state's block has one)
    float max_pose_err_ = 0.0f, max_joint_vel_ = 0.0f;
    double hold_time_ = 0.0, timeout_ = 0.0;
    bool settled_ = false;
    double settled_since_ = 0.0;
};

REGISTER_FSM(State_JointLock)
