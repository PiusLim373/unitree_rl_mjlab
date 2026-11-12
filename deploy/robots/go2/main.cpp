#include "FSM/CtrlFSM.h"
#include "FSM/State_Damping.h"
#include "FSM/State_JointLock.h"
#include "FSM/State_RLBase.h"
#include "FSM/State_Velocity.h"
#include "FSM/State_Recovery.h"
#include "FSM/State_DonkeyKick.h"
#include "FSM/State_UprightWalk.h"

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    init_fsm_state();

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Controls (Unitree's layout; PS pad in Sim2Sim: X Triangle, Y Square, Start Options):\n"
              << "  [L2 + B]   Damping, from any state\n"
              << "  [L2 + A]   Damping -> JointLock (stand up); JointLock -> LieDown (lie down,\n"
              << "             then damping by itself); Velocity -> JointLock\n"
              << "  [Start]    JointLock <-> Velocity (walk); back to JointLock only when stopped\n"
              << "  [D-pad]    in Velocity: drive forward/back/sideways at a fixed speed\n"
              << "  [R2 + Y]   DonkeyKick, [R2 + X] QuickBuck: standing still, hands back by itself\n"
              << "  [R2 + Left / Right]  SideFlip left / right: same (hold R2, then tap the D-pad)\n"
              << "  [R2 + Up / Down]  FrontFlip / BackFlip: same\n"
              << "             A skill that times out upright hands over to Velocity; a fall goes to damping\n"
              << "  [L2 + Up]  Recovery after a fall: lies down, then damping by itself\n"
              << "  [L2 + Y]   JointLock -> UprightWalk: rises on the rear legs, sticks walk\n"
              << "             in it: [L2 + A] come down to JointLock, [R2 + Left / Right] paw out / back,\n"
              << "             [R1 + Left / Right] wave / back (one arm at a time)\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

