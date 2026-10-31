// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();

        // First action before handing over (this runs in the FSM thread): the next run()
        // sends it with the new gains, never zeros or the last episode's targets.
        const auto t0 = std::chrono::steady_clock::now();
        env->reset();
        env->step();
        spdlog::info("{}: first action after {:.1f} ms", getStateString(),
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());

        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing (the first step is done)
            auto sleepTill = clock::now() + dt;
            int late = 0;
            double worst_ms = 0.0;
            auto last_warn = clock::now();

            while (policy_thread_running)
            {
                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;

                env->step();

                // A step that ends past the next deadline: count it, warn every 2 s at most.
                const auto now = clock::now();
                if (now > sleepTill) {
                    ++late;
                    worst_ms = std::max(worst_ms, std::chrono::duration<double, std::milli>(now - sleepTill).count());
                    if (now - sleepTill > dt) sleepTill = now;   // too far behind: resync, no burst
                }
                if (late > 0 && now - last_warn > std::chrono::seconds(2)) {
                    spdlog::warn("{}: {} policy steps late (worst {:.1f} ms)", getStateString(), late, worst_ms);
                    late = 0;
                    worst_ms = 0.0;
                    last_warn = now;
                }
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)
