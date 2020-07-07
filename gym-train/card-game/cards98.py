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

    def __init__(self, timeout_turn=1000, layers=(500, 500)):
        """
        self.pile_going_up = [1, 1]
        self.pile_going_down = [100, 100]
        """
        self._reset()

        os.makedirs(os.path.join("models", card_settings.MODEL_NAME), exist_ok=True)
        self.timeout_turn = timeout_turn

        self.translator = MapIndexesToNum(4, 8)
        self.batch_index, self.plot_num, self.layers = self.load_config()
        print(f"Loaded layers: {self.layers}")
        if self.layers is None:
            self.layers = layers
        else:
            if len(self.layers) == len(layers):  # update dropout values
                self.layers = [new_num if num < 1 and new_num < 1 else num for num, new_num in
                               zip(self.layers, layers)]

        self.model = self.create_model()
        self.load_weights()
        self.tensorboard = CustomTensorBoard(log_dir=f"tensorlogs/{card_settings.MODEL_NAME}-{self.plot_num}",
                                             step=self.batch_index)
        self.memory = deque(maxlen=card_settings.MEMOR_MAX_SIZE)

        'Rewards'
        self.WIN = 0
        self.SkipMove = card_settings.SKIP_MOVE
        self.GoodMove = card_settings.GOOD_MOVE
        self.EndGame = card_settings.LOST_GAME
        self.WrongMove = card_settings.INVALID_MOVE

    def _reset(self):
        self.piles = [1, 1, 100, 100]
        self.deck = random.sample(range(2, 100), 98)  # 98)
        self.hand = []
        self.move_count = 0
        self.turn = 0

        self.score = 0
        self.score_gained = 0
        self.good_moves = 0
        self.bad_moves = 0
        self.hand_ind = -1
        self.pile_ind = -1
        self.last_card_played = 0
        self.history = []
        self.hand_fill()

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

    def calculate_chance_10(self, cards, round_chance=True):
        """

        Args:
            cards:
            round_chance:

        Returns:

        """
        #
        # Check propabality of playing Card Higher or lower by 10
        #        
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
        [lower_chance, higher_chance] = self.calculate_chance_10(self.hand)

        if show_chances:
            lower_chance_row = [str(i) + '%' for i in lower_chance]  # Making text list, Adding % to number
            higher_chance_row = [str(i) + '%' for i in higher_chance]  # Making text list, Adding % to number
            hand.add_row(['Lower Card Chance'] + lower_chance_row)

        hand_with_nums = [str(i + 1) + '# ' + str(j) for i, j in enumerate(self.hand)]  # Numerated Hand
        hand.add_row(['Hand'] + hand_with_nums)

        if show_chances:
            hand.add_row(['Higher Card Chance'] + higher_chance_row)
        print(hand.draw())

    def end_condition(self, force_end=False):
        """
        Checking end conditions after current move
        Returns:
            end_game: dictionary of bool's [loss, win, end, other]
            end_bool: boolean value if game has ended
        """
        end_game = {}.fromkeys(['loss', 'win', 'timeout', 'other'], False)
        end_bool = False
        if force_end:
            end_game['other'] = True
            end_bool = True
            return end_game, end_bool

        if self.move_count > self.timeout_turn:
            end_game['timeout'] = True
            end_bool = True
            self.score_gained = self.EndGame
            return end_game, end_bool

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
            end_game['win'] = True
            end_bool = True
            self.score_gained = self.WIN
        else:
            end_game['loss'] = True
            end_bool = True
            self.score_gained = self.EndGame
        return end_game, end_bool

    def get_user_input(self):
        """
        Reading numbers from input
        Method Return:
          True:   Move
          None:   Command
          False:  Stop or Interrupts
          Second object is score feedback
        """
        self.hand_ind, self.pile_ind = -1, -1
        print('Select card and pile:')
        game_input = input()
        nums = re.findall(r'\d', game_input)

        if len(nums) == 2:
            self.hand_ind = int(nums[0]) - 1
            self.pile_ind = int(nums[1]) - 1
            return True
        else:
            game_input = game_input.split()
            for word in game_input:
                word = word.lower()

                if 'res' in word or 'new' in word:
                    self.reset()
                    return None

                elif 'end' in word or 'over' in word or 'quit' in word \
                        or 'exit' in word:
                    return False

    def hand_fill(self):
        """
        Fill Hand with cards from deck
        Hand is always 8
        """
        while len(self.hand) < 8 and len(self.deck) > 0:
            self.hand.append(self.deck[0])
            self.deck.pop(0)
        self.hand.sort()

    @staticmethod
    def input_random():
        """
        Random input generators (for testing purposes)
        """
        a = round(random.random() * 7) + 1
        b = round(random.random() * 3) + 1
        return a, b

    def main_loop(self, ai_play=True, eps=0):
        new_state = self.observation()
        train_interval = 0

        while True:

            old_state = new_state

            if card_settings.ALLOW_TRAIN and random.random() < eps:
                action = np.random.randint(0, 32)
            else:
                action = self.predict(old_state)
            move = self.translator.get_map(action)

            if not card_settings.ALLOW_TRAIN:
                self.display_table()
                print(move)
                time.sleep(0.1)

            valid = self.play_card(move)
            new_state = self.observation()
            if valid:
                end_dict, end_bool = self.end_condition()
                reward = self.score_gained
            else:
                "Move was invalid"
                end_dict, end_bool = self.end_condition(force_end=True)
                reward = self.WrongMove

            self.memory_add(old_state, new_state, action, reward, end_bool)

            train_interval += 1
            if end_bool:
                break

        if card_settings.ALLOW_TRAIN:
            self.train_model()

        if eps < 0.01:
            if self.move_count - self.turn > 1:
                print("2! " * 500)
            print(
                    f"'{card_settings.MODEL_NAME}' score: {self.score:>8.1f}    Good/Bad: {self.turn:>3} /{self.move_count - self.turn:>4}  "
                    f"eps: {eps:<7.3f} ", end='')
            if end_dict['win']:
                print("=== Win !!! ===")
            elif end_dict['timeout']:
                print("! time !")
            elif end_dict['other']:
                print("invalid")
            else:
                print("lost")

    def play_card(self, action):
        """
        Returns List Bool
        Plays Card from hand to pile.
        Checks for Valid move.
        Invalid moves return None.
        Add Turn Counter at proper moves.
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
                self.score_gained = reward
                self.turn += 1
                self.hand_fill()
                return True
            else:
                self.score += self.WrongMove
                self.score_gained = self.WrongMove
                return False

        except IndexError as ie:
            # print(f'INDEX ERROR: {ie}, {action}, {len(self.hand)}')
            self.score += self.WrongMove
            self.score_gained = self.WrongMove
            return False

    def reset(self):
        """
        Reset game
        Returns:

        """
        self._reset()
        obs = self.observation()
        return obs

    def play_game(self, load_save=False):
        """
        Start New Game or Load Save
        """
        self.reset()

        if load_save:
            file = open('data/98CardsGame_SaveFile.json', 'r')
            self.deck = json.load(file)
            file.close()
        self.main_loop(ai_play=False)

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

    def memory_add(self, old_state, new_state, action, reward, done):
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
        shuffle(train_data)

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
            #     print(new_q, ft_r)
            #     time.sleep(5)
            current_Qs[index, act] = new_q

        self.model.fit(old_states, current_Qs, verbose=0, callbacks=[self.tensorboard])

    def predict(self, state):
        """Return single action"""
        state = np.array(state).reshape(-1, *card_settings.INPUT_SHAPE)
        action = np.argmax(self.model.predict(state), axis=1)
        return action[0]

    def observation(self):
        """Return cards in deck(asumption we know play history) and in hand"""
        piles = self.conv_piles_to_array()
        hand = self.conv_hand_to_array()
        out = np.concatenate([piles, hand])
        return out


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


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    sess = tf.compat.v1.Session(config=config)

    app = GameCards98(timeout_turn=card_settings.GAME_TIMEOUT, layers=card_settings.LAYERS)
    time_start = time.time()
    EPS = iter(np.linspace(card_settings.EPS, 0, 100))

    try:
        for x in range(card_settings.GAME_NUMBER):
            if (time.time() - time_start) > card_settings.TRAIN_TIMEOUT:
                print("Train timeout")
                break

            try:
                eps = next(EPS)
            except StopIteration:
                EPS = iter(np.linspace(card_settings.EPS, 0, 50))
                eps = 0

            app.reset()
            app.main_loop(eps=eps)

            if card_settings.DEBUG:
                app.train_model()
                break
            if not card_settings.ALLOW_TRAIN:
                app.display_table()
                break

            # if not x % 100:
            #     app.save_all()
    except KeyboardInterrupt:
        if card_settings.ALLOW_TRAIN:
            app.save_all()
        print("Keyboard STOP!")

    print(f"Training end: {card_settings.MODEL_NAME}")
    print(f"Layers: {app.layers}")
    if card_settings.ALLOW_TRAIN:
        app.save_all()
