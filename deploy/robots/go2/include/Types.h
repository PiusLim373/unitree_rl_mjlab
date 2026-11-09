#pragma once

#include "unitree/dds_wrapper/robots/go2/go2.h"

// Sends /lowcmd from the FSM thread. The SDK's publisher works on its own thread while the
// FSM writes msg_, so frames tear and fail their CRC, which the motors make audible.
class LowCmd_t
{
public:
    using MsgType = unitree_go::msg::dds_::LowCmd_;

    MsgType msg_{};

    LowCmd_t() : publisher_("rt/lowcmd")
    {
        msg_.head() = {0xFE, 0xEF};
        msg_.level_flag() = 0xFF;
        for (auto & m : msg_.motor_cmd()) m.mode(1);
    }

    // Keeps the SDK's name: FSMState.h is shared with the other robots.
    void unlockAndPublish()
    {
        if (++tick_ % kTicksPerFrame) return;
        msg_.crc() = crc32_core((uint32_t*)&msg_, (sizeof(MsgType) >> 2) - 1);
        publisher_.Write(msg_, 0);
    }

private:
    static constexpr unsigned kTicksPerFrame = 2;   // Unitree's own rate

    unitree::robot::PublisherBase<MsgType> publisher_;
    unsigned tick_ = 0;
};

using LowState_t = unitree::robot::go2::subscription::LowState;