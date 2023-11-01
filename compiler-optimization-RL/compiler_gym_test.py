import gym
import compiler_gym

print(compiler_gym.COMPILER_GYM_ENVS)
env = gym.make("llvm-autophase-ic-v0")
help(env.step)
