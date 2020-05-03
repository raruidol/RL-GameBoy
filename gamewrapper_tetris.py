import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam
import gym
from pyboy import PyBoy, WindowEvent
from collections import deque


filename = 'ROMs/Tetris_World.gb'
quiet = True
pyboy = PyBoy(filename, window_type="headless" if quiet else "SDL2", window_scale=3, debug=not quiet, game_wrapper=True)
pyboy.set_emulation_speed(0)
tetris = pyboy.game_wrapper()


def game_area_changed(ga1, ga2):
    if np.array_equal(ga1, ga2):
        return True
    else:
        return False


def tetris_game_ended(ga):
    if 135 in ga:
        return True
    else:
        return False


def press_right():
    pyboy.send_input(WindowEvent.PRESS_ARROW_RIGHT)
    pyboy.send_input(WindowEvent.RELEASE_ARROW_RIGHT)


def press_left():
    pyboy.send_input(WindowEvent.PRESS_ARROW_LEFT)
    pyboy.send_input(WindowEvent.RELEASE_ARROW_LEFT)


ACTIONS = [press_right(), press_left()]


def get_state(state):
    st = np.asarray(state, dtype='float32')
    return st.reshape(1,180)


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.game_area().shape
        model.add(Dense(24, input_shape=(180, ), activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(len(ACTIONS)))
        model.summary()
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return ACTIONS[0]
        pred = self.model.predict(state)
        return np.argmax(pred[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


def main():

    tetris.start_game()

    gamma = 0.9
    epsilon = .95

    trials = 1000
    trial_len = 500

    # updateTargetNetwork = 1000
    dqn_agent = DQN(env=tetris)
    steps = []
    for trial in range(trials):
        cur_state = get_state(tetris.game_area())

        #print(cur_state)
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)

            # En cada paso recoger el nuevo estado, la reward y la comprovación de si está done la partida
            pyboy.tick()
            new_state = get_state(tetris.game_area())
            reward = tetris.score
            done = pyboy.tick()
            #new_state, reward, done, _ = env.step(action)
            #print(new_state, reward, done)

            # reward = reward if not done else -20
            #new_state = new_state.reshape(1, 2)
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()  # internally iterates default (prediction) model
            dqn_agent.target_train()  # iterates target model

            cur_state = new_state
            if done:
                break
        if step >= 199:
            print("Failed to complete in trial {}".format(trial))
            if step % 10 == 0:
                dqn_agent.save_model("trial-{}.model".format(trial))
        else:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("success.model")
            break


if __name__ == "__main__":
    main()
