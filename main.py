env_config = 'scripts/configs/IntersectionEnv/env.json'
agent_config = 'scripts/configs/IntersectionEnv/agents/DQNAgent/ego_attention_2h.json'

from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment

NUM_EPISODES = 100

env = load_environment(env_config)
agent = load_agent(agent_config, env)
evaluation = Evaluation(env, agent, num_episodes=NUM_EPISODES, display_env=False, display_agent=False)
print(f"Ready to train {agent} on {env}")

evaluation.train()
"""
# Включаем TensorBoard
from tensorboard import program
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', str(evaluation.directory)])  # Преобразуем WindowsPath в строку
url = tb.launch()"""

# @title Run the learned policy for a few episodes.
env = load_environment(env_config)
env.config["offscreen_rendering"] = True
agent = load_agent(agent_config, env)
evaluation = Evaluation(env, agent, num_episodes=30, training=False, recover=True)
evaluation.test()
