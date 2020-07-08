from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Input, concatenate
from keras import backend as backend
from keras.callbacks import TensorBoard
import tensorflow as tf

from collections import deque
from random import shuffle, sample

import texttable as tt
import numpy as np
import random  # random
import time
import json
import re  # regex
import sys
import os

import card_settings


class CustomTensorBoard(TensorBoard):
    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, step=0, **kwargs):
        super().__init__(**kwargs)
        self.step = step
        self._log_write_dir = self.log_dir
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


class GameCards98:
    """
    Piles    1: GoingUp 2: GoingUp'
             3: GoingDown 4: GoingDown'
    Input: hand_number, pile number ; Separator is not necessary'
    """

    def __init__(self, timeout_turn=1000):
        """

        Args:
            timeout_turn:
        """
        self._reset()

        self.timeout_turn = timeout_turn
        self.translator = MapIndexesToNum(4, 8)

        'Rewards'
        self.WIN = 0
        self.SkipMove = card_settings.SKIP_MOVE
        self.GoodMove = card_settings.GOOD_MOVE
        self.EndGame = card_settings.LOST_GAME
        self.InvalidMove = card_settings.INVALID_MOVE

    def reset(self):
        """
        Reset game
        Returns:
            state
        """
        self._reset()
        obs = self.observation()
        return obs

    def _reset(self):
        self.piles = [1, 1, 100, 100]
        self.deck = random.sample(range(2, 100), 98)  # 98)
        self.hand = []
        self.move_count = 0
        self.turn = 0

        self.score = 0
        self.score_gained = 0
        self.last_card_played = 0
        self.history = []
        self.hand_fill()

    def calculate_chance_10(self, cards, round_chance=True):
        """
        Check propabality of playing Card Higher or lower by 10
        Args:
            cards:
            round_chance:

        Returns:

        """
        lower_card_chance = []
        higher_card_chance = []

        if len(self.deck) > 0:
            chance = round(1 / len(self.deck) * 100, 2)
            if round_chance:
                chance = round(chance)
            chance = chance
        else:
            chance = 0

        for card in cards:
            # Checking for cards in deck -> Chance %
            # Checking for cards in hand -> 100%
            # Not Checking piles
            if card - 10 in self.deck:
                lower_card_chance.append(chance)
            # elif (card - 10 in self.piles[2:4]) or card - 10 in self.hand:
            elif card - 10 in self.hand:
                lower_card_chance.append(100)
            else:
                lower_card_chance.append(0)

            if card + 10 in self.deck:
                higher_card_chance.append(chance)
            # elif card + 10 in self.piles[0:2] or card + 10 in self.hand:
            elif card + 10 in self.hand:
                higher_card_chance.append(100)
            else:
                higher_card_chance.append(0)
        return [lower_card_chance, higher_card_chance]

    def conv_piles_to_array(self):
        out = np.array(self.piles.copy())
        out = out / 100
        return out

    def conv_dec_to_array(self):
        cards = np.zeros(98, dtype=int)
        for card_index in self.deck:
            cards[card_index - 2] = 1  # Card '2' is first on list, index 0
        return np.array(cards)

    def conv_hand_to_array(self):
        cards = np.zeros((8, 98), dtype=int)
        for index, card_index in enumerate(self.hand):
            cards[index, card_index - 2] = 1  # Card '2' is first on list, index 0
        cards = cards.ravel()
        return cards

    def check_move(self, hand_id, pile_id):
        """
        Method Checks if move is proper
        Returns True if valid
        Returns False if invalid
        Used in checking for End Conditions
        Copied from Play Card Method
        """
        response = {}.fromkeys(['invalid', 'valid', 'skip'], False)
        valid = False

        if hand_id < 0 or hand_id > 7:
            "Invalid move"

        elif pile_id < 0 or pile_id > 3:
            "Invalid move"

        elif pile_id == 0 or pile_id == 1:
            try:
                if self.hand[hand_id] > self.piles[pile_id]:
                    valid = True
                elif self.hand[hand_id] == (self.piles[pile_id] - 10):
                    valid = True
                    response['skip'] = True
            except IndexError:
                "Invalid move"

        elif pile_id == 2 or pile_id == 3:
            try:
                if self.hand[hand_id] < self.piles[pile_id]:
                    valid = True
                elif self.hand[hand_id] == (self.piles[pile_id] + 10):
                    valid = True
                    response['skip'] = True
            except IndexError:
                "Invalid move"

        if valid:
            response['valid'] = True
            return True, response
        else:
            response['invalid'] = True
            return False, response

    def display_table(self, show_chances=False, show_deck=False):
        """
        Showing Table.
        Showing Hand.
        Showing Chances of next Cards.
        """
        print('\n' + '=' * 5, 'Turn'.center(8), '=', self.move_count)
        print('=' * 5, 'Score'.center(8), '=', self.score)
        if show_deck:
            print('Deck (cheating) :', self.deck)

        piles = tt.Texttable()
        piles.add_row(['↑ Pile ↑', '1# ' + str(self.piles[0]), '2# ' + str(self.piles[1])])
        piles.add_row(['↓ Pile ↓', '3# ' + str(self.piles[2]), '4# ' + str(self.piles[3])])
        print(piles.draw())

        hand = tt.Texttable()
        lower_chance, higher_chance = self.calculate_chance_10(self.hand)

        if show_chances:
            lower_chance_row = [str(i) + '%' for i in lower_chance]  # Making text list, Adding % to number
            higher_chance_row = [str(i) + '%' for i in higher_chance]  # Making text list, Adding % to number
            hand.add_row(['Lower Card Chance'] + lower_chance_row)

        hand_with_nums = [str(i + 1) + '# ' + str(j) for i, j in enumerate(self.hand)]  # Numerated Hand
        hand.add_row(['Hand'] + hand_with_nums)

        if show_chances:
            hand.add_row(['Higher Card Chance'] + higher_chance_row)
        print(hand.draw())

    def end_condition(self):
        """
        Checking end conditions after current move
            Returns:

            end_game: Dict[str, bool],  keys = [loss, win, end, other]

            end_bool: boolean value if game has ended

            reward:
        """
        info = {}.fromkeys(['loss', 'win', 'timeout', 'other'], False)
        end_bool = False
        reward = None
        if self.move_count > self.timeout_turn:
            info['timeout'] = True
            end_bool = True
            return end_bool, info, self.EndGame

        next_move = None
        for hand_id in range(8):
            if next_move:
                break

            for pile_id in range(4):
                next_move, info = self.check_move(hand_id, pile_id)
                if next_move:
                    break

        if next_move:
            pass
        elif len(self.hand) == 0 and len(self.deck) == 0:
            info['win'] = True
            end_bool = True
            reward = self.WIN
        else:
            info['loss'] = True
            end_bool = True
            reward = self.EndGame
        return end_bool, info, reward

    def hand_fill(self):
        """
        Fill Hand with cards from deck
        Hand is always 8
        """
        while len(self.hand) < 8 and len(self.deck) > 0:
            self.hand.append(self.deck[0])
            self.deck.pop(0)
        self.hand.sort()

    def step(self, action):
        """
        Play 1 card
        Args:
            action: Tuple(int, int)
        Returns:
            reward, new_state, done, info
        """
        valid, reward = self._play_card(action)
        if valid:
            done, info, end_reward = self.end_condition()
            if done:
                reward = end_reward
        else:
            info = {}.fromkeys(['loss', 'win', 'timeout', 'other'], False)
            info['other'] = True
            done = True

        new_state = self.observation()
        return reward, new_state, done, info

    def _play_card(self, action):
        """

        Args:
            action:

        Returns:

        """
        pile_id, hand_id = action
        self.move_count += 1
        self.score_gained = 0  # reset value

        try:
            valid, info = self.check_move(hand_id, pile_id)
            if valid:
                card = self.hand[hand_id]
                self.piles[pile_id] = card
                self.hand.pop(hand_id)

                if info['skip']:
                    reward = self.SkipMove
                else:
                    reward = self.GoodMove
                self.score += reward
                self.turn += 1
                self.hand_fill()
                return True, reward

            else:
                self.score += self.InvalidMove
                reward = self.InvalidMove
                return False, reward

        except IndexError as ie:
            print(f'INDEX ERROR: {ie}, {action}, {len(self.hand)}')
            self.score += self.InvalidMove
            reward = self.InvalidMove
            return False, reward

    def observation(self):
        """Return cards in deck(asumption we know play history) and in hand"""
        piles = self.conv_piles_to_array()
        hand = self.conv_hand_to_array()
        out = np.concatenate([piles, hand])
        return out


class Agent:
    def __init__(self, layers):
        os.makedirs(os.path.join("models", card_settings.MODEL_NAME), exist_ok=True)

        self.batch_index, self.plot_num, self.layers = self.load_config()
        if self.layers is None:
            self.layers = layers
        else:
            if len(self.layers) == len(layers):  # update dropout values
                self.layers = [new_num if num < 1 and new_num < 1 else num for num, new_num in
                               zip(self.layers, layers)]
        print(f"Layers: {self.layers}")

        self.model = self.create_model()
        self.load_weights()
        self.memory = deque(maxlen=card_settings.MEMOR_MAX_SIZE)
        self.tensorboard = CustomTensorBoard(log_dir=f"tensorlogs/{card_settings.MODEL_NAME}-{self.plot_num}",
                                             step=self.batch_index)

    def load_config(self):
        batch_index = self.load_batch()
        plot_num = self.load_plot_num()
        if batch_index > card_settings.GRAPH_CUT_AT:
            batch_index = 0
            plot_num += 1
        layers = self.load_layers()
        return batch_index, plot_num, layers

    def load_layers(self):
        if os.path.isfile(f"models/{card_settings.MODEL_NAME}/layers.npy"):
            layers = np.load(f"models/{card_settings.MODEL_NAME}/layers.npy", allow_pickle=True)
            return layers
        else:
            return None

    def load_weights(self):
        if os.path.isfile(f"models/{card_settings.MODEL_NAME}/model"):
            print("Loaded weights")
            while True:
                try:
                    self.model.load_weights(f"models/{card_settings.MODEL_NAME}/model")
                    break
                except OSError as oe:
                    print(f"Oe: {oe}")
                    time.sleep(0.2)
            return True
        else:
            return False

    def load_batch(self):
        if os.path.isfile(f"models/{card_settings.MODEL_NAME}/batch.npy"):
            batch = np.load(f"models/{card_settings.MODEL_NAME}/batch.npy", allow_pickle=True)
            print(batch)
            return batch
        else:
            return 0

    def load_plot_num(self):
        if os.path.isfile(f"models/{card_settings.MODEL_NAME}/plot.npy"):
            num = np.load(f"models/{card_settings.MODEL_NAME}/plot.npy", allow_pickle=True)
            return num
        else:
            return 0

    def save_all(self):
        try:
            while True:
                try:
                    self.model.save_weights(f"models/{card_settings.MODEL_NAME}/model")
                    break
                except OSError:
                    time.sleep(0.2)
            while True:
                try:
                    np.save(f"models/{card_settings.MODEL_NAME}/batch", self.batch_index)
                    break
                except OSError:
                    time.sleep(0.2)
            while True:
                try:
                    np.save(f"models/{card_settings.MODEL_NAME}/plot", self.plot_num)
                    break
                except OSError:
                    time.sleep(0.2)
            while True:
                try:
                    np.save(f"models/{card_settings.MODEL_NAME}/layers", self.layers)
                    break
                except OSError:
                    time.sleep(0.2)
            print("Saved all.")
            return True
        except KeyboardInterrupt:
            print("Keyboard Interrupt: saving again")
            self.save_all()
            sys.exit(0)

    def create_model(self):
        input_layer = Input(shape=card_settings.INPUT_SHAPE)
        print(f"Creating model: {card_settings.MODEL_NAME}: {self.layers}")
        last = input_layer
        for num in self.layers:
            if 0 < num <= 1:
                drop = Dropout(num)(last)
                last = drop
            elif num > 1:
                num = int(num)
                dense = Dense(num, activation='relu')(last)
                last = dense
            else:
                raise ValueError(f"This values is below 0: {num}")
        value = Dense(32, activation='linear')(last)
        model = Model(inputs=input_layer, outputs=value)

        model.compile(optimizer=Adam(learning_rate=card_settings.ALPHA), loss='mse', metrics=['accuracy'])
        with open(f"models/{card_settings.MODEL_NAME}/summary.txt", 'w') as file:
            model.summary(print_fn=lambda x: file.write(x + '\n'))
        return model

    def add_memmory(self, old_state, new_state, action, reward, done):
        self.memory.append((old_state, new_state, action, reward, done))

    def train_model(self):
        train_data = list(self.memory)

        if len(train_data) < card_settings.MIN_BATCH_SIZE:
            return None
        else:
            self.batch_index += 2

        if card_settings.CLEAR_MEMORY:
            self.memory.clear()

        if len(train_data) > card_settings.MAX_BATCH_SIZE:
            train_data = sample(train_data, card_settings.MAX_BATCH_SIZE)

        old_states = []
        new_states = []
        actions = []
        rewards = []
        dones = []
        for index, (old_st, new_st, act, rw, dn) in enumerate(train_data):
            old_states.append(old_st)
            new_states.append(new_st)
            actions.append(act)
            rewards.append(rw)
            dones.append(dn)

        # old_pile, old_hand = np.array(old_states)[:, 0], np.array(old_states)[:, 1]
        # new_pile, new_hand = np.array(new_states)[:, 0], np.array(new_states)[:, 1]
        # old_pile = old_pile.reshape(-1, 4)
        # old_hand = old_hand.reshape(-1, 8)
        # new_pile = new_pile.reshape(-1, 4)
        # new_hand = new_hand.reshape(-1, 8)

        old_states = np.array(old_states)
        new_states = np.array(new_states)

        current_Qs = self.model.predict(old_states)
        future_maxQ = np.max(self.model.predict(new_states), axis=1)
        for index, (act, rew, dn, ft_r) in enumerate(zip(actions, rewards, dones, future_maxQ)):
            new_q = rew + ft_r * card_settings.DISCOUNT * int(not dn)
            current_Qs[index, act] = new_q

        self.model.fit(old_states, current_Qs, verbose=0, callbacks=[self.tensorboard])

    def predict(self, state):
        """Return single action"""
        state = np.array(state)
        actions = np.argmax(self.model.predict(state), axis=1)
        return actions


class MapIndexesToNum:
    """
    Maps 2-Dimensional arrays or higher to 1 number index and reverse
    """

    def __init__(self, *dimensions):
        self.shape = list(dimensions)
        self.dims = len(self.shape)

        ind = 1
        for dim in dimensions:
            ind *= dim
        self.max_ind = ind - 1

    def get_num(self, *indexes):
        if type(indexes[0]) is tuple:
            indexes = indexes[0]

        if len(indexes) != self.dims:
            raise ValueError("Dimensions number does not match")

        num = 0
        multi = 1
        for ind, size in zip(indexes, self.shape):
            if ind >= size:
                raise ValueError(f"Index must be lower than size: {ind} >= {size}")
            num += ind * multi
            multi *= size
        return num

    def get_map(self, input_index):
        """Returns tuple of indexes matching index"""
        if 0 <= input_index <= self.max_ind:
            num = int(input_index)
            out = []
            for size in self.shape:
                cur_in = num % size
                num = num // size
                out.append(cur_in)
            return tuple(out)

        else:
            raise ValueError("Index beyond range")


def train_model():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.compat.v1.Session(config=config)
    try:
        episode_offset = np.load(f"models/{card_settings.MODEL_NAME}/last-episode-num.npy", allow_pickle=True)
    except FileNotFoundError:
        episode_offset = 0
    stats = {
            "episode": [],
            "eps": [],
            "score": [],
            "good_moves": []}

    agent = Agent(layers=card_settings.LAYERS)
    trans = MapIndexesToNum(4, 8)
    time_start = time.time()
    time_save = time.time()
    EPS = iter(np.linspace(card_settings.EPS, 0, 100))
    try:
        for episode in range(episode_offset, card_settings.GAME_NUMBER + episode_offset):

            if (time.time() - time_start) > card_settings.TRAIN_TIMEOUT:
                print("Train timeout")
                break
            try:
                eps = next(EPS)
            except StopIteration:
                EPS = iter(np.linspace(card_settings.EPS, 0, 50))
                eps = 0

            Games = []  # Close screen
            States = []
            for loop_ind in range(card_settings.SIM_COUNT):
                game = GameCards98(timeout_turn=card_settings.GAME_TIMEOUT)
                state = game.reset()
                Games.append(game)
                States.append(state)

            Scores = [0] * len(Games)
            step = 0
            All_score = []
            All_steps = []
            while len(Games):
                step += 1
                Old_states = np.array(States)
                if eps > np.random.random():
                    Actions = np.random.randint(0, card_settings.ACTION_SPACE, size=(len(Old_states)))
                else:
                    Actions = agent.predict(Old_states)
                Dones = []
                Rewards = []
                States = []

                for g_index, game in enumerate(Games):
                    move = trans.get_map(Actions[g_index])
                    reward, state, done, info = game.step(action=move)
                    Rewards.append(reward)
                    Scores[g_index] += reward
                    Dones.append(done)
                    States.append(state)

                if card_settings.ALLOW_TRAIN:
                    for old_s, act, rew, n_st, dn in zip(Old_states, Actions, Rewards, States, Dones):
                        agent.add_memmory(old_s, n_st, act, rew, dn)
                    if card_settings.STEP_TRAIN:
                        agent.train_model()

                for ind_d in range(len(Games) - 1, -1, -1):
                    if Dones[ind_d]:

                        All_score.append(Scores[ind_d])
                        All_steps.append(Games[ind_d].move_count)

                        stats['episode'].append(episode + episode_offset)
                        stats['eps'].append(eps)
                        stats['score'].append(Scores[ind_d])
                        stats['good_moves'].append(step)

                        Scores.pop(ind_d)
                        Games.pop(ind_d)
                        States.pop(ind_d)

            if card_settings.ALLOW_TRAIN and not episode % card_settings.TRAIN_EVERY:
                agent.train_model()
            episode += 1
            if eps < 0.1:
                print(f"'{card_settings.MODEL_NAME}' "
                      f"avg-score: {np.mean(All_score):>6.2f}, "
                      f"worst-game: {np.min(All_score):>6.1f}, "
                      f"avg-good-move: {np.mean(All_steps):>4.1f}, "
                      f"best-moves: {np.max(All_steps):>4}, "

                      f"eps: {eps:<5.2f}")
            if time.time() - card_settings.SAVE_INTERVAL > time_save:
                time_save = time.time()
                agent.save_all()

    except KeyboardInterrupt:
        if card_settings.ALLOW_TRAIN:
            agent.save_all()
        print("Keyboard STOP!")

    print(f"Training end: {card_settings.MODEL_NAME}")
    print(f"Layers: {agent.layers}")
    if card_settings.ALLOW_TRAIN:
        agent.save_all()
        np.save(f"models/{card_settings.MODEL_NAME}/last-episode-num.npy", episode)


def show_game():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.compat.v1.Session(config=config)

    agent = Agent(layers=card_settings.LAYERS)
    trans = MapIndexesToNum(4, 8)
    game = GameCards98(timeout_turn=card_settings.GAME_TIMEOUT)
    states = [game.reset()]
    done = False

    while not done:
        game.display_table()
        action = agent.predict(states)[0]
        move = trans.get_map(action)
        pile, hand = move
        print(f"Move: {hand + 1} -> {pile + 1}")
        rew, new_state, done, info = game.step(move)
    print(info)


if __name__ == '__main__':
    if card_settings.ALLOW_TRAIN:
        train_model()
        print(f"Learning rate: {card_settings.ALPHA}")
    else:
        show_game()
