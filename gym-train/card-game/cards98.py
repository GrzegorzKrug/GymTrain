from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Input
from keras import backend as backend
from keras.callbacks import TensorBoard
import tensorflow as tf

from collections import deque
from random import shuffle

import texttable as tt
import numpy as np
import random  # random
import time
import json
import re  # regex
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
        self.pile_going_up = [1, 1]
        self.pile_going_down = [100, 100]
        """
        self._reset()

        os.makedirs(os.path.join("models", card_settings.MODEL_NAME), exist_ok=True)
        self.timeout_turn = timeout_turn

        self.translator = MapIndexesToNum(4, 8)
        self.model = self.create_model()
        self.load_weights()
        self.batch_index = self.load_batch()

        self.tensorboard = CustomTensorBoard(log_dir=f"tensorlogs/{card_settings.MODEL_NAME}", step=self.batch_index)
        'Rewards'
        self.GoodMove = 1
        self.WrongMove = -10
        self.SkipMove = 9  # This move + 8 cards in hand
        self.EndGame = 100

    def _reset(self):
        self.piles = [1, 1, 100, 100]
        self.deck = random.sample(range(2, 100), 98)  # 98)
        self.hand = []
        self.move_count = 0
        self.turn = 0

        self.score = 0
        self.score_gained = 0
        self.hand_ind = -1
        self.pile_ind = -1
        self.last_card_played = 0
        self.history = []
        self.memory = deque(maxlen=10000)

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

    def conv_dec_to_array(self):
        cards = np.zeros(98, dtype=int)
        for card_index in self.deck:
            cards[card_index - 2] = 1  # Card '2' is first on list, index 0
        return np.array(cards)

    def conv_hand_to_array(self):
        cards = np.zeros(98, dtype=int)
        for card_index in self.hand:
            cards[card_index - 2] = 1  # Card '2' is first on list, index 0
        return np.array(cards)

    def check_move(self, hand_id, pile_id):
        """
        Method Checks if move is proper
        Returns True if valid
        Returns False if invalid
        Used in checking for End Conditions
        Copied from Play Card Method
        """
        if hand_id < 0 or hand_id > 7:
            # print('Error: Invalid hand index')
            return False

        elif pile_id < 0 or pile_id > 3:
            # print('Error: Invalid pile index')
            return False

        elif pile_id == 0 or pile_id == 1:
            try:
                if self.hand[hand_id] > self.piles[pile_id] or \
                        self.hand[hand_id] == (self.piles[pile_id] - 10):
                    return True
                else:
                    return False
            except IndexError:
                return False

        elif pile_id == 2 or pile_id == 3:
            try:
                if self.hand[hand_id] < self.piles[pile_id] or \
                        self.hand[hand_id] == (self.piles[pile_id] + 10):
                    return True
                else:
                    return False
            except IndexError:
                return False

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

    def end_condition(self):
        """
        Checking end conditions after current move
        Returns:
            end_game: dictionary of bool's [loss, win, end, other]
            end_bool: boolean value if game has ended
        """
        end_game = {}.fromkeys(['loss', 'win', 'timeout', 'other'], False)
        end_bool = False

        if self.move_count > self.timeout_turn:
            end_game['timeout'] = True
            end_bool = True
            return end_game, end_bool

        next_move = None
        for hand_id in range(8):
            if next_move:
                break

            for pile_id in range(4):
                next_move = self.check_move(hand_id, pile_id)
                if next_move:
                    break

        if next_move:
            pass
        elif len(self.hand) == 0 and len(self.deck) == 0:
            end_game['win'] = True
            end_bool = True
        else:
            end_game['loss'] = True
            end_bool = True

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
        while len(self.hand) < 7 and len(self.deck) > 0:
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

    def main_loop(self, ai_play=True):
        new_state = self.observation()
        train_interval = 0
        while True:
            train_interval += 1
            if not train_interval % 20:
                self.train_model()

            old_state = new_state
            self.hand_fill()
            if ai_play:
                if random.random() < card_settings.EPS:
                    action = np.random.randint(0, 32)
                else:
                    action = self.ai_predict(old_state)
                move = self.translator.get_map(action)
            else:
                self.display_table()
                move = self.get_user_input()  # Replace user input with NN

            self.play_card(move)
            new_state = self.observation()
            reward = self.score_gained
            end_dict, end_bool = self.end_condition()
            self.memory_add(old_state, new_state, action, reward, end_bool)

            if end_bool:
                break
        # if end_dict['loss']:
        #     print("You have lost")
        if end_dict['win']:
            print("=== Win !!! ===")
        # elif end_dict['timeout']:
        #     print("Timeout!")
        # elif end_dict['other']:
        #     print("Other reason of end")
        print(f"Good moves: {self.turn}")

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
            if hand_id < 0 or hand_id > 7:  # Invalid Hand index
                print('Error: Invalid hand index')
                self.score += self.WrongMove
                self.score_gained = self.WrongMove
                return False

            elif pile_id < 0 or pile_id > 3:  # Invalid Pile index
                print(f'Error: Invalid pile index: {pile_id}')
                self.score += self.WrongMove
                self.score_gained = self.WrongMove
                return False

            elif pile_id == 0 or pile_id == 1:  # Rising Piles
                if self.hand[hand_id] > self.piles[pile_id]:  # Played card is higher
                    self.piles[pile_id] = self.hand[hand_id]
                    self.hand.pop(hand_id)
                    self.score += self.GoodMove
                    self.score_gained = self.GoodMove
                    self.turn += 1
                    return True

                elif self.hand[hand_id] == (self.piles[pile_id] - 10):  # Played card is lower by 10
                    self.piles[pile_id] = self.hand[hand_id]
                    self.hand.pop(hand_id)
                    self.score += self.SkipMove
                    self.score_gained = self.SkipMove
                    self.turn += 1
                    return True
                else:  # Invalid Move
                    # print('Not valid move!')
                    self.score += self.WrongMove
                    self.score_gained = self.WrongMove
                    return False

            elif pile_id == 2 or pile_id == 3:  # Lowering Piles
                if self.hand[hand_id] < self.piles[pile_id]:  # Played card is lower
                    self.piles[pile_id] = self.hand[hand_id]
                    self.hand.pop(hand_id)
                    self.score += self.GoodMove
                    self.score_gained = self.GoodMove
                    self.turn += 1
                    return True

                elif self.hand[hand_id] == (self.piles[pile_id] + 10):  # Played card is higher by 10
                    self.piles[pile_id] = self.hand[hand_id]
                    self.hand.pop(hand_id)
                    self.score += self.SkipMove
                    self.score_gained = self.SkipMove
                    return True

                else:  # Invalid Move
                    self.score += self.WrongMove
                    self.score_gained = self.WrongMove
                    return False
            else:
                input('Impossible! How did u get here?!')

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

    def ai_predict(self, state):
        """Return single action"""
        state = np.array(state).reshape(-1, 196)
        action = np.argmax(self.model.predict(state), axis=1)
        return action[0]

    def load_weights(self):
        if os.path.isfile(f"models/{card_settings.MODEL_NAME}/model"):
            print("Loaded weights")
            while True:
                try:
                    self.model.load_weights(f"models/{card_settings.MODEL_NAME}/model")
                    break
                except OSError:
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

    def save_model(self):
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
        return True

    def create_model(self):
        input_layer = Input(shape=(196,))
        dense1 = Dense(512, activation='relu')(input_layer)
        dense2 = Dense(512, activation='relu')(dense1)
        value = Dense(32, activation='linear')(dense2)
        model = Model(inputs=input_layer, outputs=value)

        model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=['accuracy'])

        return model

    def memory_add(self, old_state, new_state, action, reward, done):
        self.memory.append((old_state, new_state, action, reward, done))

    def train_model(self):
        train_data = list(self.memory)
        if len(train_data) < 1:
            return None
        else:
            self.batch_index += 2
        self.memory.clear()
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

        # targets = np.zeros((len(actions), 32))
        new_states = np.array(new_states)
        old_states = np.array(old_states)

        current_Qs = self.model.predict(old_states)
        future_maxQ = np.argmax(self.model.predict(new_states), axis=1)
        for index, (act, rew, dn, ft_r) in enumerate(zip(actions, rewards, dones, future_maxQ)):
            current_Qs[index, act] = rew + ft_r * card_settings.DISCOUNT * (not dn)

        self.model.fit(old_states, current_Qs, verbose=0, callbacks=[self.tensorboard])

    def observation(self):
        """Return cards in deck(asumption we know play history) and in hand"""
        a = self.conv_dec_to_array()
        b = self.conv_hand_to_array()
        out = np.concatenate([a, b])
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

    app = GameCards98(timeout_turn=card_settings.GAME_TIMEOUT)
    for x in range(10000):
        app.reset()
        app.main_loop()
        if not x % 25:
            app.save_model()
    app.save_model()
