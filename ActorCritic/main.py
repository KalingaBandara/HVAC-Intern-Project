import numpy as numpy 
from actor_critic import Agent
from utils import plot_learning_curve

import energyplus.ooep as ooep
import energyplus.ooep.ems
import energyplus.ooep.addons.state
from energyplus.dataset.basic import dataset as epds


if __name__ == '__main__':

    env = energyplus.ooep.ems.Environment().__enter__()

    sm_env = ooep.addons.state.StateMachine(env)
    sm_stepf = sm_env.step_function(
    dict(event_name='begin_zone_timestep_after_init_heat_balance')
)
    sm_env.run(
    '--output-directory', 'build3/demo-eplus',
    '--weather', f'./SGP_Singapore_486980_IWEC.epw',
    f'./new1ch.idf',
    verbose=True,
)



    agent = Agent(alpha=1e-5, n_actions=4)
    n_games = 1800

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
    
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        
        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, score_history, figure_file)
