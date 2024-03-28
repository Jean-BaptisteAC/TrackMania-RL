from src.environment.TMNFEnv import TrackmaniaEnv

from stable_baselines3 import PPO
import keyboard

if __name__ == "__main__":

    env = TrackmaniaEnv(observation_space="image", dimension_reduction=6)

    model_type = "PPO"
    models_dir = "models/" + model_type
    model_watch_name = "PPO_Training_Flat_Dataset_env_fix"
    model_step = "638k"


    model_path = f"{models_dir}/{model_watch_name}/{model_step}"
    model_to_watch = PPO.load(model_path, env=env)
    obs, _ = env.reset()
    while True:

        action, _ = model_to_watch.predict(obs)
        obs, reward, done, _, info = env.step(action)


        if info["checkpoint_time"] is not False:
            print(info["checkpoint_time"])

        try:
            if keyboard.is_pressed("q"):
                print("Interrupt")
                break
        except:
            pass

    env.close()