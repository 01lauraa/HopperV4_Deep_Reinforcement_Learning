
import gymnasium as gym
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore", category=UserWarning) 

class ChangeMassWrapper(gym.Wrapper):
    def __init__(self, env, torso_mass):
        super().__init__(env)  
        self.torso_mass = torso_mass
        self.env.model.body_mass[1] = self.torso_mass


def train_model(seed, torso_mass):
    n_envs = 4
    env = make_vec_env('Hopper-v4', n_envs=n_envs, seed=seed, vec_env_cls=SubprocVecEnv,
                       wrapper_class=ChangeMassWrapper, wrapper_kwargs={'torso_mass': torso_mass})
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    tensorboard_path = f"./ppo_hopper_tensorboard/seed_{seed}mass{torso_mass}"
    # new_logger = configure(tensorboard_path, ["stdout", "csv", "tensorboard"])

    model = PPO("MlpPolicy", env, verbose=1, ent_coef = 0.01)

    # model.set_logger(new_logger)
    model.learn(total_timesteps=1e6)  

    model.save(f"results/ppo_hopper_seed_{seed}mass{torso_mass}")
    env.save(f"results/ppo_hopper_vec_normalize_seed_{seed}mass{torso_mass}.pkl")

    env.close()

def evaluate_all_models():
    seeds = [1, 2, 3]
    torso_masses = [3, 6, 9]
    test_masses = [3, 4, 5, 6, 7, 8, 9]
    results = []
    for seed in seeds:
        for mass in torso_masses:
            print(f"Evaluating seed {seed} and torso mass {mass}kg")
            model = PPO.load(f"results_completed/ppo_hopper_seed_{seed}mass{mass}.zip")
            for testmass in test_masses:
                env = make_vec_env('Hopper-v4', n_envs=1, seed=seed, vec_env_cls=SubprocVecEnv,
                                   wrapper_class=ChangeMassWrapper, wrapper_kwargs={'torso_mass': testmass})
                env = VecNormalize.load(f"results_completed/ppo_hopper_vec_normalize_seed_{seed}mass{mass}.pkl", env)
                
                env.training = False 
                env.norm_reward = False 

                mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
               #print(f"Torso mass {testmass}kg: {mean_reward} +/- {std_reward}")
                env.close()
                results.append((seed, mass, testmass, mean_reward, std_reward))
    df = pd.DataFrame(results, columns=["seed", "train_mass", "test_mass", "mean_reward", "std_reward"])
    df.to_csv("results_completed/Q3_results.csv")

    return df

def plot_results(df: pd.DataFrame):
    fig, ax = plt.subplots(ncols=3, figsize=(10, 5))

    for mass in [3,6,9]:
        df_mass = df[df["train_mass"] == mass]
        df_mass = df_mass.groupby(['train_mass', 'test_mass']).aggregate({'mean_reward': 'mean', 'std_reward': 'mean'}).reset_index()
        
        plt.subplot(1, 3, mass//3)
        plt.plot(df_mass['test_mass'], df_mass['mean_reward'], label='Mean Reward')
        plt.xlabel('Torso Mass')
        plt.legend([f'm= {mass}'])
        plt.ylabel('Performance')
        plt.xticks(df_mass['test_mass'])
        plt.fill_between(df_mass['test_mass'], df_mass['mean_reward'] - df_mass['std_reward'], df_mass['mean_reward'] + df_mass['std_reward'], alpha=0.2)
        print(df_mass)
        #assert False
    plt.tight_layout()
    plt.savefig("results_completed/Q3_results.png")

def main():
    #Q1
    #train_model(1, 3)

    #Q2
    # seeds = [1, 2, 3]
    # torso_masses = [3, 6, 9]
    
    # for seed in seeds:
    #     for mass in torso_masses:
    #         print(f"Training with seed {seed} and torso mass {mass}kg")
    #         train_model(seed, mass)

    #Q3
    df = evaluate_all_models()
    df = pd.read_csv("results_completed/Q3_results.csv")
    plot_results(df)



if __name__ == '__main__':  
    main()

