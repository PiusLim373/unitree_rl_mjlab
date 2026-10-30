// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"

class State_Damping : public FSMState
{
public:
    State_Damping(int state, std::string state_string = "Damping") 
    : FSMState(state, state_string) 
    {
        auto motor_mode = param::config["FSM"]["Damping"]["mode"];
        if(motor_mode.IsDefined())
        {
            auto values = motor_mode.as<std::vector<int>>();
            for(int i(0); i<values.size(); ++i)
            {
                lowcmd->msg_.motor_cmd()[i].mode() = values[i];
            }
        }
    } 

    void enter()
    {
        // set gain
        static auto kd = param::config["FSM"]["Damping"]["kd"].as<std::vector<float>>();
        for(int i(0); i < kd.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = 0;
            motor.kd() = kd[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }
    }

    void run()
    {
        for(int i(0); i < lowcmd->msg_.motor_cmd().size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();
        }
    }
};

REGISTER_FSM(State_Damping)
