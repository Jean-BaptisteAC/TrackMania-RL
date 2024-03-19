from src.environment.TMNFEnv import TrackmaniaEnv

if __name__ == "__main__":
    env =  TrackmaniaEnv(action_space="controller", observation_space="image")

    env.simthread.iface.execute_command("warp 0")
    env.simthread.iface.execute_command("save_state")
    env.simthread.iface.execute_command("load_state")