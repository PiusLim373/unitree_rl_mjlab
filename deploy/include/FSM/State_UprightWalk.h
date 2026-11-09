#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>

#include "FSM/State_RLBase.h"
#include "FSM/upright_math.h"
#include "isaaclab/envs/mdp/observations/observations.h"

// UprightWalk: one policy (src/tasks/upright, loaded like Velocity from its own
// policy_dir) that rises from JointLock, stands and walks on the rear legs on cmd_vel,
// balances with a soft paw out, and comes back down. Started from JointLock only.
//
// Next to cmd_vel the policy reads the up/down command and the hand flag, which this
// state sets as training's UprightCommand does (mdp/commands.py):
//   come_down: an arm that is out comes back first, then the up command goes to 0
//   paw_left / paw_right: a tap puts that paw out (while upright, both arms in), another
//     tap puts it back; no time limit
//   wave_left / wave_right: the same for a wave
//   cmd_vel reads 0 while an arm is out or on its way back, and while coming down
//
// An arm out takes over its front leg as training's HandJointPositionAction does, and the
// policy's targets for that leg are ignored:
//   paw: over paw.ramp_time its motors blend to soft gains aimed at paw.pose, with gravity
//     compensation. Back: the soft arm moves to the rest pose over paw.back_time, then its
//     gains return over paw.stiffen_time.
//   wave: the leg's own gains and gravity compensation, the hip swinging around
//     wave.center; in and out over wave.ramp_time.
//   tipped out of the stance (tip_exit), an arm out or on its way back goes straight back
//   to the policy, which then rises again (training's rise; a slow return let it fall).
//
// Checks after the gamepad transitions (L2 + B Damping), every FSM tick (1 kHz), first
// true wins:
//   - refused: on entry the robot wasn't level, still and in JointLock's stand pose
//     (start_check); the policy never starts -> back to JointLock
//   - fell: nose or roll beyond fall_check (training's fell limits) -> Damping
//   - down: after come_down, level and still for hold_time -> JointLock
//   - timeout: not down down_check.timeout after the up command went to 0 -> Damping
class State_UprightWalk : public State_RLBase
{
public:
    State_UprightWalk(int state_mode, std::string state_string)
    : State_RLBase(state_mode, state_string)
    {
        auto cfg = param::config["FSM"][state_string];
        come_down_ = compile(cfg["come_down"]);
        paw_left_ = compile(cfg["paw_left"]);
        paw_right_ = compile(cfg["paw_right"]);
        wave_left_ = compile(cfg["wave_left"]);
        wave_right_ = compile(cfg["wave_right"]);

        auto start = cfg["start_check"];
        cos_start_tilt_ = std::cos(start["max_tilt"].as<float>());
        start_gyro_ = start["max_gyro"].as<float>();
        start_joint_vel_ = start["max_joint_vel"].as<float>();
        start_pose_err_ = start["max_pose_err"].as<float>();
        stand_pose_ = start["stand_pose"].as<std::vector<float>>();  // SDK motor order

        auto fall = cfg["fall_check"];
        min_nose_ = fall["min_nose"].as<float>();
        max_nose_ = fall["max_nose"].as<float>();
        max_roll_ = fall["max_roll"].as<float>();

        auto down = cfg["down_check"];
        cos_down_tilt_ = std::cos(down["max_tilt"].as<float>());
        down_joint_vel_ = down["max_joint_vel"].as<float>();
        down_hold_ = std::chrono::duration<double>(down["hold_time"].as<double>());
        down_timeout_ = std::chrono::duration<double>(down["timeout"].as<double>());

        auto paw = cfg["paw"];
        paw_ramp_ = paw["ramp_time"].as<float>();
        paw_back_ = paw["back_time"].as<float>();
        paw_stiffen_ = paw["stiffen_time"].as<float>();
        paw_min_nose_ = paw["min_nose"].as<float>();
        paw_pose_ = paw["pose"].as<std::vector<float>>();
        paw_kp_ = paw["kp"].as<std::vector<float>>();
        paw_kd_ = paw["kd"].as<std::vector<float>>();

        auto wave = cfg["wave"];
        wave_ramp_ = wave["ramp_time"].as<float>();
        wave_center_ = wave["center"].as<std::vector<float>>();
        wave_hip_amp_ = wave["hip_amp"].as<float>();
        wave_freq_ = wave["freq"].as<float>();

        auto tip = cfg["tip_exit"];
        tip_min_nose_ = tip["min_nose"].as<float>();
        tip_max_roll_ = tip["max_roll"].as<float>();

        const int joint_lock = FSMStringMap.right.at("JointLock");
        const int damping = FSMStringMap.right.at("Damping");
        this->registered_checks.emplace_back([this]()->bool{ return refused_; }, joint_lock);
        this->registered_checks.emplace_back(
            [this]()->bool{
                if (refused_ || !fell_now()) return false;
                spdlog::warn("{}: fell over (nose {:.0f} deg, roll {:.0f} deg), switching to damping "
                             "(press L2 + Up to recover).", this->getStateString(),
                             nose() * 180.0f / static_cast<float>(M_PI), roll() * 180.0f / static_cast<float>(M_PI));
                return true;
            },
            damping);
        this->registered_checks.emplace_back(
            [this]()->bool{
                if (refused_ || !down_long_enough()) return false;
                spdlog::info("{}: down on four legs; back to JointLock.", this->getStateString());
                return true;
            },
            joint_lock);
        this->registered_checks.emplace_back(
            [this]()->bool{
                if (refused_ || !going_down_ || clock::now() - down_at_ < down_timeout_) return false;
                spdlog::warn("{}: not down after {:.1f} s, switching to damping.",
                             this->getStateString(), down_timeout_.count());
                return true;
            },
            damping);
    }

    void enter()
    {
        refused_ = !ready();
        if (refused_)
        {
            spdlog::warn("{}: refused, the robot isn't standing level and still in JointLock's pose.",
                         getStateString());
            return;
        }
        down_requested_ = going_down_ = settled_ = waving_ = false;
        hand_side_ = ramp_side_ = 0;
        ramp_ = pose_ = 0.0f;
        last_tick_ = clock::now();
        isaaclab::mdp::upright_up_cmd = 1.0f;
        isaaclab::mdp::upright_hand_side = 0;
        isaaclab::mdp::upright_cmd_vel_on = true;
        // A stick held while pressing the combo doesn't drive it.
        isaaclab::mdp::hold_velocity_commands = true;
        spdlog::info("{}: go.", getStateString());
        State_RLBase::enter();
    }

    // Refused: command nothing, so the previous state's last command holds for the
    // one tick before the hand-back.
    void run()
    {
        if (refused_) return;
        update_commands();
        State_RLBase::run();
        apply_arm();
    }

private:
    using clock = std::chrono::steady_clock;
    using Button = std::function<bool(unitree::common::UnitreeJoystick &)>;

    static Button compile(const YAML::Node & condition)
    {
        unitree::common::dsl::Parser parser(condition.as<std::string>());
        return unitree::common::dsl::Compile(*parser.Parse());
    }

    // These read LowState directly: this FSM thread refreshes it every tick (pre_run),
    // while the policy's articulation data belongs to the 50 Hz thread.

    Eigen::Quaternionf quat() const
    {
        const auto & q = lowstate->msg_.imu_state().quaternion();  // w, x, y, z
        return upright::to_quat({q[0], q[1], q[2], q[3]});
    }

    float nose() const { return upright::nose_angle(quat()); }
    float roll() const { return upright::roll_angle(quat()); }

    float cos_tilt() const
    {
        const auto & q = lowstate->msg_.imu_state().quaternion();
        return 1.0f - 2.0f * (q[1] * q[1] + q[2] * q[2]);
    }

    float max_joint_speed() const
    {
        float v = 0.0f;
        for (int i = 0; i < 12; ++i) v = std::max(v, std::fabs(lowstate->msg_.motor_state()[i].dq()));
        return v;
    }

    bool ready() const
    {
        if (cos_tilt() < cos_start_tilt_) return false;
        const auto & w = lowstate->msg_.imu_state().gyroscope();
        if (std::sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]) > start_gyro_) return false;
        for (int i = 0; i < 12; ++i)
        {
            if (std::fabs(lowstate->msg_.motor_state()[i].q() - stand_pose_[i]) > start_pose_err_) return false;
        }
        return max_joint_speed() <= start_joint_vel_;
    }

    bool fell_now() const
    {
        const float n = nose();
        return n < min_nose_ || n > max_nose_ || std::fabs(roll()) > max_roll_;
    }

    // True once level and still has held for hold_time after come_down.
    bool down_long_enough()
    {
        if (!going_down_ || cos_tilt() < cos_down_tilt_ || max_joint_speed() > down_joint_vel_)
        {
            settled_ = false;
            return false;
        }
        const auto now = clock::now();
        if (!settled_)
        {
            settled_ = true;
            settled_since_ = now;
        }
        return now - settled_since_ >= down_hold_;
    }

    // One arm at a time: the same combo again puts it back; any other waits until it is in.
    void toggle_arm(int side, bool wave, clock::time_point now)
    {
        const std::string name = std::string(side == 1 ? "left" : "right") + (wave ? " wave" : " paw");
        if (hand_side_ == side && waving_ == wave)
        {
            hand_side_ = 0;
            spdlog::info("{}: {} back.", getStateString(), name);
        }
        else if (hand_side_ != 0 || ramp_ > 0.0f)
        {
            spdlog::warn("{}: {} ignored, an arm is still out or on its way back.", getStateString(), name);
        }
        else if (nose() < paw_min_nose_)
        {
            spdlog::warn("{}: {} ignored, not upright yet.", getStateString(), name);
        }
        else
        {
            hand_side_ = ramp_side_ = side;
            waving_ = wave;
            wave_t0_ = now;
            spdlog::info("{}: {} out.", getStateString(), name);
        }
    }

    void update_commands()
    {
        auto & js = lowstate->joystick;
        const auto now = clock::now();
        const float dt = std::min(std::chrono::duration<float>(now - last_tick_).count(), 0.05f);
        last_tick_ = now;

        if (!down_requested_ && come_down_(js))
        {
            down_requested_ = true;
            spdlog::info("{}: coming down{}.", getStateString(),
                         hand_side_ != 0 || ramp_ > 0.0f ? " once the arm is back" : "");
            hand_side_ = 0;
        }
        if (!down_requested_)
        {
            if (paw_left_(js)) toggle_arm(1, false, now);
            else if (paw_right_(js)) toggle_arm(2, false, now);
            else if (wave_left_(js)) toggle_arm(1, true, now);
            else if (wave_right_(js)) toggle_arm(2, true, now);
        }
        if (ramp_ > 0.0f && (nose() < tip_min_nose_ || std::fabs(roll()) > tip_max_roll_))
        {
            spdlog::warn("{}: tipped (nose {:.0f} deg, roll {:.0f} deg), arm back to the policy at once.",
                         getStateString(), nose() * 180.0f / static_cast<float>(M_PI),
                         roll() * 180.0f / static_cast<float>(M_PI));
            hand_side_ = 0;
            ramp_ = pose_ = 0.0f;
        }

        // As training: out at once. Back: a paw's soft arm reaches the rest pose first, then
        // the gains return; a wave just ramps out.
        const bool out = hand_side_ != 0;
        pose_ = out ? 1.0f : (waving_ ? 0.0f : std::max(pose_ - dt / paw_back_, 0.0f));
        const float target = out ? 1.0f : (pose_ > 0.0f ? ramp_ : 0.0f);
        const float step = dt / (waving_ ? wave_ramp_ : (out ? paw_ramp_ : paw_stiffen_));
        ramp_ += std::clamp(target - ramp_, -step, step);
        if (ramp_ <= 0.0f && ramp_side_ != 0)
        {
            ramp_side_ = 0;
            if (!down_requested_) spdlog::info("{}: arm in, walking again.", getStateString());
        }

        if (down_requested_ && !going_down_ && ramp_ <= 0.0f)
        {
            going_down_ = true;
            down_at_ = now;
        }
        isaaclab::mdp::upright_up_cmd = going_down_ ? 0.0f : 1.0f;
        isaaclab::mdp::upright_hand_side = hand_side_;
        isaaclab::mdp::upright_cmd_vel_on = !down_requested_ && hand_side_ == 0 && ramp_ <= 0.0f;
    }

    // After State_RLBase::run: the arm's front leg blends from the policy to the paw or wave.
    void apply_arm()
    {
        const auto body = quat();
        const float t = std::chrono::duration<float>(clock::now() - wave_t0_).count();
        const float swing = wave_hip_amp_ * std::sin(2.0f * static_cast<float>(M_PI) * wave_freq_ * t);
        for (int side = 1; side <= 2; ++side)
        {
            const int first = side == 1 ? 3 : 0;  // SDK order: FR, FL
            const int leg = side == 1 ? 0 : 3;    // policy order: FL, FR
            const float w = ramp_side_ == side ? ramp_ : 0.0f;
            const float soft = waving_ ? 0.0f : w;  // a wave keeps the leg's gains
            const float sign = side == 1 ? 1.0f : -1.0f;
            Eigen::Vector3f q;
            for (int k = 0; k < 3; ++k) q[k] = lowstate->msg_.motor_state()[first + k].q();
            const Eigen::Vector3f hold = upright::paw_gravity_torque(sign, q, body);
            for (int k = 0; k < 3; ++k)
            {
                auto & motor = lowcmd->msg_.motor_cmd()[first + k];
                const float mirror = k == 0 ? sign : 1.0f;  // hip abduction mirrors
                const float rest = env->robot->data.default_joint_pos[leg + k];
                const float pose = waving_ ? mirror * (wave_center_[k] + (k == 0 ? swing : 0.0f))
                                           : rest + pose_ * (mirror * paw_pose_[k] - rest);
                motor.q() = motor.q() * (1.0f - w) + pose * w;
                motor.kp() = env->robot->data.joint_stiffness[first + k] * (1.0f - soft) + paw_kp_[k] * soft;
                motor.kd() = env->robot->data.joint_damping[first + k] * (1.0f - soft) + paw_kd_[k] * soft;
                motor.tau() = hold[k] * w;
            }
        }
    }

    Button come_down_, paw_left_, paw_right_, wave_left_, wave_right_;

    float cos_start_tilt_, start_gyro_, start_joint_vel_, start_pose_err_;
    std::vector<float> stand_pose_;
    float min_nose_, max_nose_, max_roll_;
    float cos_down_tilt_, down_joint_vel_;
    std::chrono::duration<double> down_hold_, down_timeout_;
    float paw_ramp_, paw_back_, paw_stiffen_, paw_min_nose_;
    std::vector<float> paw_pose_, paw_kp_, paw_kd_;
    float wave_ramp_, wave_hip_amp_, wave_freq_;
    std::vector<float> wave_center_;
    float tip_min_nose_, tip_max_roll_;

    bool refused_ = false, down_requested_ = false, going_down_ = false, settled_ = false;
    bool waving_ = false;  // the arm out (or last out) is a wave
    int hand_side_ = 0;    // 0 none, 1 left, 2 right
    int ramp_side_ = 0;    // the arm the ramp belongs to, until it is back in
    float ramp_ = 0.0f;    // 0 the policy's leg, 1 the arm's
    float pose_ = 0.0f;    // a paw's target: 1 its pose, 0 the rest pose (training's hand_pose)
    clock::time_point last_tick_, down_at_, settled_since_, wave_t0_;
};

REGISTER_FSM(State_UprightWalk)
