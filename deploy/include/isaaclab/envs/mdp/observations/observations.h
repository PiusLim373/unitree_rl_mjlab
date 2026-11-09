// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <atomic>
#include <deque>

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

// Optional keys under commands/base_velocity in deploy.yaml (absent = raw sticks):
//   deadzone:        stick values up to it read 0; the rest is rescaled so full stick stays 1.
//   stick_scale:     full stick = these (forward, sideways, turn), default 1.
//   dpad_lin_vel_x:  while D-pad up/down is held, command exactly +/-value forward
//   dpad_lin_vel_y:  while D-pad left/right is held, command exactly +/-value sideways
//                    (left = +y). Both combine (diagonals); the sticks are ignored while any
//                    driving D-pad direction is held, and the D-pad doesn't drive while L1, L2,
//                    R1 or R2 is held (button combos such as L2 + up).
// hold_velocity_commands: set by State_Velocity when a skill hands over to it (a skill's
// D-pad combo may still be held); the command reads 0 until the sticks are centred and
// the D-pad released once.
inline std::atomic<bool> hold_velocity_commands{false};

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3, 0.0f);
    auto & joystick = env->robot->data.joystick;

    if (hold_velocity_commands)
    {
        constexpr float kCentred = 0.1f;  // State_Velocity's stop_check max_stick
        const bool neutral = std::fabs(joystick->lx()) <= kCentred && std::fabs(joystick->ly()) <= kCentred
            && std::fabs(joystick->rx()) <= kCentred && !joystick->up.pressed && !joystick->down.pressed
            && !joystick->left.pressed && !joystick->right.pressed;
        if (!neutral) return obs;
        hold_velocity_commands = false;
    }

    const auto cmd_cfg = env->cfg["commands"]["base_velocity"];
    const auto cfg = cmd_cfg["ranges"];
    const float dpad_vx = cmd_cfg["dpad_lin_vel_x"].IsDefined() ? cmd_cfg["dpad_lin_vel_x"].as<float>() : 0.0f;
    const float dpad_vy = cmd_cfg["dpad_lin_vel_y"].IsDefined() ? cmd_cfg["dpad_lin_vel_y"].as<float>() : 0.0f;
    const float deadzone = cmd_cfg["deadzone"].IsDefined() ? cmd_cfg["deadzone"].as<float>() : 0.0f;

    const bool combo = joystick->LT.pressed || joystick->RT.pressed || joystick->LB.pressed || joystick->RB.pressed;
    const bool dpad_x = !combo && dpad_vx > 0.0f && (joystick->up.pressed || joystick->down.pressed);
    const bool dpad_y = !combo && dpad_vy > 0.0f && (joystick->left.pressed || joystick->right.pressed);
    if (dpad_x || dpad_y)
    {
        if (dpad_x) obs[0] = joystick->up.pressed ? dpad_vx : -dpad_vx;
        if (dpad_y) obs[1] = joystick->left.pressed ? dpad_vy : -dpad_vy;
    }
    else
    {
        auto shape = [deadzone](float x) {
            float mag = std::fabs(x);
            if (mag <= deadzone) return 0.0f;
            return std::copysign((mag - deadzone) / (1.0f - deadzone), x);
        };
        const auto scale = cmd_cfg["stick_scale"].IsDefined() ? cmd_cfg["stick_scale"].as<std::vector<float>>()
                                                              : std::vector<float>{1.0f, 1.0f, 1.0f};
        obs[0] = shape(joystick->ly()) * scale[0];
        obs[1] = -shape(joystick->lx()) * scale[1];
        obs[2] = -shape(joystick->rx()) * scale[2];
    }

    obs[0] = std::clamp(obs[0], cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(obs[1], cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(obs[2], cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());

    return obs;
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    auto cmd = isaaclab::mdp::velocity_commands(env, params);
    float cmd_norm = std::sqrt(
        cmd[0] * cmd[0] +
        cmd[1] * cmd[1] +
        cmd[2] * cmd[2]
    );

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);

    if (cmd_norm < 0.1f)
    {
        obs[0] = 0.0f;
        obs[1] = 0.0f;
    }

    return obs;
}

// DonkeyKick timer (training: src/tasks/kick kick_phase): time since the state was
// entered as a fraction of params.duration, held at 1. The policy thread resets
// episode_length to 0 on entry and counts it up before each observation, so step k
// (from 1) is at (k - 1) * step_dt, as in training (first observation at t = 0).
REGISTER_OBSERVATION(kick_phase)
{
    float duration = params["duration"].as<float>();
    float t = static_cast<float>(env->episode_length - 1) * env->step_dt;
    return std::vector<float>{std::clamp(t / duration, 0.0f, 1.0f)};
}

// UprightWalk's inputs next to cmd_vel (training: src/tasks/upright/mdp/upright.py), set
// by State_UprightWalk in the FSM thread, read here in the policy thread.
inline std::atomic<float> upright_up_cmd{1.0f};   // 1 = stand upright, 0 = down on four legs
inline std::atomic<int> upright_hand_side{0};     // 0 none, 1 left, 2 right
inline std::atomic<bool> upright_cmd_vel_on{true};  // false: cmd_vel reads 0 (paw out, coming down)

REGISTER_OBSERVATION(upright_commands)
{
    if (!upright_cmd_vel_on) return std::vector<float>(3, 0.0f);
    return velocity_commands(env, params);
}

// The gait clock while walking, (0, 0) at zero cmd_vel (training's STAND_THRESHOLD). As
// kick_phase, step k (from 1) is at (k - 1) * step_dt.
REGISTER_OBSERVATION(upright_phase)
{
    const auto cmd = upright_commands(env, params);
    if (std::hypot(cmd[0], cmd[1]) + std::fabs(cmd[2]) <= 0.1f) return std::vector<float>(2, 0.0f);
    const float period = params["period"].as<float>();
    const float t = static_cast<float>(env->episode_length - 1) * env->step_dt;
    const float phase = std::fmod(t, period) / period;
    return {std::sin(2.0f * static_cast<float>(M_PI) * phase), std::cos(2.0f * static_cast<float>(M_PI) * phase)};
}

REGISTER_OBSERVATION(upright_up)
{
    return std::vector<float>{upright_up_cmd};
}

REGISTER_OBSERVATION(upright_hand)
{
    const int side = upright_hand_side;
    return std::vector<float>{side == 1 ? 1.0f : 0.0f, side == 2 ? 1.0f : 0.0f};
}

// The last second in coarse bins (training: mdp/commands.py slow_history). Each policy step
// adds a frame of gravity, gyro, rear joint positions (from default) and speeds; out come
// `bins` means of `bin_steps` frames, oldest first. An episode's first call (the reset, then
// its first step) fills it with the current frame, as training does for a robot just reset.
REGISTER_OBSERVATION(upright_slow_history)
{
    const int bins = params["bins"].as<int>();
    const int bin_steps = params["bin_steps"].as<int>();
    auto & data = env->robot->data;
    std::vector<float> frame(data.projected_gravity_b.data(), data.projected_gravity_b.data() + 3);
    frame.insert(frame.end(), data.root_ang_vel_b.data(), data.root_ang_vel_b.data() + 3);
    const int n = data.joint_pos.size();
    constexpr int kRear = 6;  // policy order FL, FR, RL, RR: the rear legs are the last six
    for (int i = n - kRear; i < n; ++i) frame.push_back(data.joint_pos[i] - data.default_joint_pos[i]);
    for (int i = n - kRear; i < n; ++i) frame.push_back(data.joint_vel[i]);

    static std::deque<std::vector<float>> frames;
    const size_t steps = static_cast<size_t>(bins * bin_steps);
    if (env->episode_length <= 1 || frames.size() != steps)
    {
        frames.assign(steps, frame);
    }
    else
    {
        frames.pop_front();
        frames.push_back(frame);
    }

    std::vector<float> out(bins * frame.size(), 0.0f);
    for (size_t s = 0; s < steps; ++s)
    {
        for (size_t k = 0; k < frame.size(); ++k) out[(s / bin_steps) * frame.size() + k] += frames[s][k] / bin_steps;
    }
    return out;
}

}
}