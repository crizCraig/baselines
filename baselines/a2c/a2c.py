import os
import os.path as osp
from logging.handlers import RotatingFileHandler

import gym
import time
import joblib
import logging
import numpy as np
import tensorflow as tf
from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import wrap_deepmind

from baselines.a2c.utils import discount_with_dones, cat_entropy_softmax
from baselines.a2c.utils import Scheduler, make_path, find_trainable_variables
from baselines.a2c.policies import CnnPolicy
from baselines.a2c.utils import cat_entropy, mse

from keras.backend.tensorflow_backend import sparse_categorical_crossentropy, categorical_crossentropy

SHOULD_PRINT_TF = False


def tp(name, tensor, summarize=100):
    if SHOULD_PRINT_TF:
        tensor = tf.Print(tensor, [name, tensor], summarize=summarize)
    return tensor


class Model(object):

    def __init__(self, policy, ob_space, ac_space, nenvs, nsteps, nstack, num_procs,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=num_procs,
                                inter_op_parallelism_threads=num_procs)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        nact = ac_space.n
        nbatch = nenvs*nsteps

        A = tf.placeholder(tf.int32, [nbatch])
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        step_model = policy(sess, ob_space, ac_space, nenvs, 1, nstack, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenvs, nsteps, nstack, reuse=True)

        advantage_local = tf.identity(ADV)
        advantage_local = tp('advantage_local', advantage_local)

        # logits = 1 - pi if ADV < 0, else pi
        pi = train_model.pi

        sign = tf.sign(advantage_local)
        sign = tf.expand_dims(sign, 1)
        sign = tf.tile(sign, [1, tf.shape(pi)[1]])
        is_neg_reward = sign * (sign - 1) / 2
        is_pos_reward = 1 - is_neg_reward  # Should be just sign now...
        reversed_pi = 1 - pi
        is_neg_reward = tp('is_neg_reward', is_neg_reward)
        is_pos_reward = tp('is_pos_reward', is_pos_reward)

        corrective_probs = is_pos_reward * pi - is_neg_reward * reversed_pi
        # corrective_probs = sign * (sign + pi) - sign * (sign + 1) / 2

        # logits = pi
        # corrective_probs = tf.Print(corrective_probs, ['corrective_probs', corrective_probs], summarize=100)


        # logits = s * (s + p) - tf.tile(tf.reshape(s * (s + 1) / 2., [1, -1]), [tf.shape(s)[0], 1])

        if 'CSQ' in os.environ:
            local_action = tf.identity(A)
            one_hot_action = tf.one_hot(local_action, depth=tf.shape(pi)[-1])
            # one_hot_action = tf.Print(one_hot_action, ['one_hot_action', one_hot_action], summarize=18)
            neglogpac = categorical_crossentropy(target=one_hot_action, output=corrective_probs)
            # neglogpac = categorical_crossentropy(target=one_hot_action, output=pi)
        else:
            if 'CSQ_min' in os.environ:
                logits = pi
                logit_max = tf.reduce_max(logits)

                logit_max = tp('logit_max', logit_max)

                logit_min = tf.reduce_min(logits)

                logit_min = tp('logit_min', logit_min)

                reversed_logits = logit_max - logits + logit_min
                reversed_logits = tp('reversed_logits', reversed_logits)

                logits = logits * is_pos_reward - reversed_logits * is_neg_reward
                pi = tp('pi', pi)
            else:
                logits = pi
            logits = tp('logitssssssssss', logits)
            neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=A)

        neglogpac = tp('neglogpac', neglogpac)

        pg_loss = tf.reduce_mean(advantage_local * neglogpac)
        # pg_loss = tf.Print(pg_loss, ['pg_loss', pg_loss])
        vf_loss = tf.reduce_mean(mse(tf.squeeze(train_model.vf), R))
        if 'CSQ' in os.environ:
            entropy = tf.reduce_mean(cat_entropy_softmax(pi))
            # ent_coef *= 100
        else:
            entropy = tf.reduce_mean(cat_entropy(pi))

        # entropy = tf.Print(entropy, ['entropy', entropy], summarize=18)

        # loss = pg_loss + vf_loss * vf_coef
        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        loss = tp('loss', loss)

        with tf.control_dependencies([tf.assert_less(loss, 1e11)]):
            loss = tp('loss', loss)
            loss = tf.identity(loss) * 2.
            loss = tf.identity(loss) / 2.

        with tf.control_dependencies([tf.assert_less(R, 1e11)]):
            rewards_local = tf.identity(R)
            rewards_local = tp('rewards_local', rewards_local)
            rewards_local *= 2
            rewards_local /= 2
            loss = loss * rewards_local
            loss = loss / rewards_local

        params = find_trainable_variables("model")
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)
        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values, advs):
            for step in range(len(obs)):
                cur_lr = lr.value()
            td_map = {train_model.X:obs, A:actions, ADV:advs, R:rewards, LR:cur_lr}
            if states != []:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        def save(save_path):
            ps = sess.run(params)
            make_path(save_path)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            ps = sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess)

class Runner(object):

    def __init__(self, env, model, nsteps=5, nstack=4, gamma=0.99):
        self.env = env
        self.model = model
        nh, nw, nc = env.observation_space.shape
        nenv = env.num_envs
        self.batch_ob_shape = (nenv*nsteps, nh, nw, nc*nstack)
        self.obs = np.zeros((nenv, nh, nw, nc*nstack), dtype=np.uint8)
        self.nc = nc
        obs = env.reset()
        self.update_obs(obs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def update_obs(self, obs):
        # Do frame-stacking here instead of the FrameStack wrapper to reduce
        # IPC overhead
        self.obs = np.roll(self.obs, shift=-self.nc, axis=3)
        self.obs[:, :, :, -self.nc:] = obs

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_advs = [],[],[],[],[],[]  # Minibatch
        mb_states = self.states
        for n in range(self.nsteps):
            actions, values, aprobs, states = self.model.step(self.obs, self.states, self.dones)
            chosen_probs = np.take(aprobs, actions)
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)
            obs, rewards, dones, _ = self.env.step(actions)
            corrected_rewards = []
            corrected_advs = []
            advs = rewards - values
            if 'MIS_ADV' in os.environ:
                corrected_rewards = rewards
                for adv in advs:
                    if adv < 0:
                        corrected_advs.append(adv * (1 + chosen_probs[i] / (1 - chosen_probs[i])))
                    else:
                        corrected_advs.append(adv)
            else:
                for i, reward in enumerate(rewards):
                    if 'SCALE_ALL_REWARDS' in os.environ:
                        corrected_rewards.append(reward * 1.8)
                    else:
                        if reward < 0:
                                # TODO: Normalize rewards for games other than Pong
                                corrected_rewards.append(reward * (1 + chosen_probs[i] / (1 - chosen_probs[i])))
                                # corrected_rewards.append(reward * (1 + chosen_probs[i] / (1 - chosen_probs[i])))  # Mistake importance scaling
                        else:
                            corrected_rewards.append(reward)

            self.states = states
            self.dones = dones
            for n, done in enumerate(dones):
                if done:
                    self.obs[n] = self.obs[n]*0
            self.update_obs(obs)
            corrected_rewards = rewards if corrected_rewards == [] else corrected_rewards
            corrected_advs = corrected_advs or advs
            assert(len(rewards) == len(corrected_rewards))
            assert(len(advs) == len(corrected_advs))
            mb_rewards.append(corrected_rewards)
            mb_advs.append(corrected_advs)
        mb_dones.append(self.dones)
        #batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
        mb_masks = mb_dones[:, :-1]
        mb_dones = mb_dones[:, 1:]
        last_values = self.model.value(self.obs, self.states, self.dones).tolist()
        #discount/bootstrap off value fn
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards+[value], dones+[0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)
            mb_rewards[n] = rewards
        mb_rewards = mb_rewards.flatten()
        mb_actions = mb_actions.flatten()
        mb_values = mb_values.flatten()
        mb_masks = mb_masks.flatten()
        return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, mb_advs

def learn(policy, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):
    tf.reset_default_graph()
    set_global_seeds(seed)

    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    num_procs = len(env.remotes) # HACK
    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack, num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
    runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

    nbatch = nenvs*nsteps
    tstart = time.time()
    for update in range(1, total_timesteps//nbatch+1):
        obs, states, rewards, masks, actions, values, advs = runner.run()
        policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values, advs)
        nseconds = time.time()-tstart
        fps = int((update*nbatch)/nseconds)
        if update % log_interval == 0 or update == 1:
            ev = explained_variance(values, rewards)
            logger.record_tabular("nupdates", update)
            logger.record_tabular("total_timesteps", update*nbatch)
            logger.record_tabular("fps", fps)
            logger.record_tabular("policy_entropy", float(policy_entropy))
            logger.record_tabular("value_loss", float(value_loss))
            logger.record_tabular("explained_variance", float(ev))
            logger.dump_tabular()
    env.close()

if __name__ == '__main__':
    main()