from policies import base_policy as bp
import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path
import os.path
import copy

GAMMA = 0.999

BATCH_SIZE = 50
NUM_OF_BATCHES = 100

LEARNING_RATE = 0.001
EPSILON = 0

DB_SIZE = 10000
INPUT = "strategy_with_feature_1"

EMPTY_VAL = 0
PLAYER1_ID = 1
PLAYER2_ID = 2

ROWS = 6
COLS = 7
WIN_MASK = np.ones(4)
ACTIONS = [0, 1, 2, 3, 4, 5, 6]


class Policy302163050(bp.Policy):

    def cast_string_args(self, policy_args):
        policy_args['save_to'] = str(policy_args['save_to']) if 'save_to' in policy_args else 'policy302163050.model.pkl'
        policy_args['load_from'] = str(policy_args['load_from']) if 'load_from' in policy_args else 'policy302163050.model.pkl'
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['batch_size'] = int(policy_args['batch_size']) if 'batch_size' in policy_args else BATCH_SIZE
        policy_args['num_of_batches'] = int(
            policy_args['num_of_batches']) if 'num_of_batches' in policy_args else NUM_OF_BATCHES
        policy_args['learning_rate'] = float(
            policy_args['learning_rate']) if 'learning_rate' in policy_args else LEARNING_RATE
        policy_args['epsilon'] = float(
            policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
        return policy_args

    def init_run(self):
        # load model
        try:
            load_from = str(Path(os.path.dirname(os.path.abspath(__file__))).parent) + "/models/" + self.load_from
            with open(load_from, 'rb') as archive:
                [weights, biases, self.epsilon] = pickle.load(archive)



            self.log("Model found", 'STATUS')


        except:
            self.log("Model not found, initializing random weights.", 'STATUS')

        # initialize neural network
        self.hidden_layers = [50, 50, 50]
        self.session = tf.Session()
        self.input_shape, self.input_flat_shape = self.get_input_dimensions()
        self.nn = NeuralNetwork(self.hidden_layers, self.input_flat_shape, session=self.session,
                                load_from=self.load_from)

        # initialize data base
        self.db = Database(state_dim=self.input_shape)

        # initialize learning parameters
        self.actions = tf.placeholder(tf.int32, (None,), "actions")
        self.q_values = self.nn.take(self.actions)
        self.q_estimation = tf.placeholder(tf.float32, (None,), name="q_estimation")
        self.loss = tf.reduce_mean((self.q_estimation - self.q_values) ** 2)
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        # update database
        converted_prev_state, converted_new_state = self.convert_boards_representations(prev_state, new_state)
        self.db.add_item(converted_prev_state, converted_new_state, prev_action, reward, done(new_state, reward))

        # train in batches
        from_idx = 0
        for i in range(self.num_of_batches):
            batch, real_batch_size = self.db.take_sample(from_idx, self.batch_size)
            if real_batch_size == 0:
                break
            second_states = batch.s2
            shape = second_states.shape
            if INPUT== "hot":
                inputs = (batch.s1).reshape(shape[0], shape[1] * shape[2]*shape[3])
                second_states = second_states.reshape(shape[0], shape[1] * shape[2]*shape[3])
            else:
                inputs = (batch.s1).reshape(shape[0], shape[1] * shape[2])
                second_states = second_states.reshape(shape[0], shape[1] * shape[2])

            v = self.nn.session.run(self.nn.output_max, feed_dict={self.nn.input: second_states})
            q = batch.r + (~batch.done) * self.gamma * v
            feed_dict = {
                self.nn.input: inputs,
                self.actions: batch.a,
                self.q_estimation: q
            }
            self.nn.session.run(self.train_op, feed_dict=feed_dict)

            from_idx += real_batch_size
            if from_idx > self.db.num_of_items:
                break

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):

        # update database
        converted_prev_state, converted_new_state = self.convert_boards_representations(prev_state, new_state)
        self.db.add_item(converted_prev_state, converted_new_state, prev_action, reward, done(new_state, reward))

        legal_actions = get_legal_moves(new_state)
        if self.mode=="train" and np.random.uniform() < self.epsilon:
            action = np.random.choice(legal_actions)
        else:
            out = self.nn.session.run(self.nn.output,
                                      feed_dict={self.nn.input: converted_new_state.reshape(1, self.input_flat_shape)})[0]
            legal_tup = np.asarray([(i, out[i]) for i in legal_actions])
            idx = np.argmax(legal_tup[:, 1])
            action = int(legal_tup[idx][0])

        return action

    def save_model(self):
        ret= [[self.nn.session.run(w) for w in self.nn.weights_iter()],
                [self.nn.session.run(b) for b in self.nn.biases_iter()], self.epsilon], self.save_to
        return ret

    def convert_boards_representations(self, prev_state, new_state):

        if INPUT == "strategy":
            converted_prev_state = self.convert_board_to_strategy_vectors(prev_state)
            converted_new_state = self.convert_board_to_strategy_vectors(new_state)
        elif INPUT == "strategy_with_feature_1":
            converted_prev_state = self.convert_board_to_strategy_vectors_with_feature1(prev_state)
            converted_new_state = self.convert_board_to_strategy_vectors_with_feature1(new_state)
        elif INPUT == "strategy_with_feature_2":
            converted_prev_state = self.convert_board_to_strategy_vectors_with_feature2(prev_state)
            converted_new_state = self.convert_board_to_strategy_vectors_with_feature2(new_state)
        elif INPUT == "hot":
            converted_prev_state = self.convert_board_to_one_hot(prev_state)
            converted_new_state = self.convert_board_to_one_hot(new_state)
        elif INPUT == "negative_positive":
            converted_prev_state = self.convert_board_representation_negative_positive(prev_state)
            converted_new_state = self.convert_board_representation_negative_positive(new_state)
        elif INPUT == "original":
            converted_prev_state = prev_state
            converted_new_state = new_state
        elif INPUT == "strategy_depth2_with_feature_1":
            converted_prev_state = self.convert_board_to_strategy_vectors_depth2_with_feature1(prev_state)
            converted_new_state = self.convert_board_to_strategy_vectors_depth2_with_feature1(new_state)
        elif INPUT == "strategy_depth2_with_feature_2":
            converted_prev_state = self.convert_board_to_strategy_vectors_depth2_with_feature2(prev_state)
            converted_new_state = self.convert_board_to_strategy_vectors_depth2_with_feature2(new_state)
        elif INPUT == "strategy_with_negative_positive":
            converted_prev_state = self.convert_board_to_strategy_with_negative_positive(prev_state)
            converted_new_state = self.convert_board_to_strategy_with_negative_positive(new_state)
        elif INPUT == "strategy_with_feature_3":
            converted_prev_state = self.convert_board_to_strategy_vectors_with_feature3(prev_state)
            converted_new_state = self.convert_board_to_strategy_vectors_with_feature3(new_state)
        return converted_prev_state, converted_new_state

    def get_input_dimensions(self):
        if INPUT == "strategy":
            input_shape = (2, COLS)
            input_flat_shape = 2 * COLS
        elif INPUT == "strategy_with_feature_1" or INPUT == "strategy_with_feature_2" or INPUT=="strategy_with_feature_3":
            input_shape = (2, COLS + 1)
            input_flat_shape = 2 * (COLS + 1)
        elif INPUT == "strategy_depth2_with_feature_1" or INPUT == "strategy_depth2_with_feature_2":
            input_shape = (4, COLS + 1)
            input_flat_shape = 4 * (COLS + 1)
        elif INPUT == "hot":
            input_shape = (2, ROWS, COLS)
            input_flat_shape = 2 * ROWS * COLS
        elif INPUT == "negative_positive" or INPUT == "original":
            input_shape = (ROWS, COLS)
            input_flat_shape = ROWS * COLS
        elif INPUT == "strategy_with_negative_positive":
            input_shape = (ROWS + 2, COLS )
            input_flat_shape = (ROWS + 2) * (COLS)

        return input_shape, input_flat_shape

    def convert_board_representation_negative_positive(self, board):
        if board is None:
            return
        if self.id == 1:
            other_player_id = 2
        else:
            other_player_id = 1
        new_board = copy.deepcopy(board)
        new_board[new_board == other_player_id] = -1
        new_board[new_board == self.id] = 1
        return new_board

    def convert_board_to_one_hot(self, board):
        if board is None:
            return
        if self.id == 1:
            other_player_id = 2
        else:
            other_player_id = 1
        new_board = np.zeros((2, board.shape[0], board.shape[1]))
        new_board[0][board == self.id] = 1
        new_board[1][board == other_player_id] = 1
        return new_board

    def convert_board_to_strategy_vectors(self, board):
        if board is None:
            return
        if self.id == 1:
            other_player_id = 2
        else:
            other_player_id = 1
        new_board = np.zeros((2, COLS))
        legal_actions = get_legal_moves(board)
        for col in legal_actions:
            if check_for_win(make_move(board, col, self.id), self.id, col):
                new_board[0][col] = 1
            if check_for_win(make_move(board, col, other_player_id), other_player_id, col):
                new_board[1][col] = 1
        return new_board

    def convert_board_to_strategy_vectors_with_feature1(self, board):
        if board is None:
            return
        if self.id == 1:
            other_player_id = 2
        else:
            other_player_id = 1
        new_board = np.zeros((2, COLS + 1))
        legal_actions = get_legal_moves(board)
        for col in legal_actions:
            if check_for_win(make_move(board, col, self.id), self.id, col):
                new_board[0][col] = 1
                new_board[0][COLS] = 1
            if check_for_win(make_move(board, col, other_player_id), other_player_id, col):
                new_board[1][col] = 1
                new_board[1][COLS] = 1
        return new_board

    def convert_board_to_strategy_vectors_with_feature2(self, board):
        if board is None:
            return
        if self.id == 1:
            other_player_id = 2
        else:
            other_player_id = 1
        new_board = np.zeros((2, COLS + 1))
        legal_actions = get_legal_moves(board)
        for col in legal_actions:
            if check_for_win(make_move(board, col, self.id), self.id, col):
                new_board[0][col] = 1
            if check_for_win(make_move(board, col, other_player_id), other_player_id, col):
                new_board[1][col] = 1
        if board[ROWS-1,int(COLS/2)]==0:
            new_board[0][int(COLS/2)]= 1
            new_board[0][COLS] = 1
        return new_board
    def convert_board_to_strategy_vectors_with_feature3(self, board):
        if board is None:
            return
        if self.id == 1:
            other_player_id = 2
        else:
            other_player_id = 1
        new_board = np.zeros((2, COLS + 1))
        legal_actions = get_legal_moves(board)
        for col in legal_actions:
            if check_for_win(make_move(board, col, self.id), self.id, col):
                new_board[0][col] = 1
                # new_board[0][COLS] = 1
            if check_for_win(make_move(board, col, other_player_id), other_player_id, col):
                new_board[1][col] = 1
                # new_board[1][COLS] = 1
        if board[ROWS-1,int(COLS/2)]==0:
            new_board[0][int(COLS/2)]= 1
            new_board[0][COLS] = 1
        for row in [0,1,5,6]:

            if board[ROWS-1,row]==0:
                new_board[0][row]= -1
        return new_board

    def convert_board_to_strategy_with_negative_positive(self, board):
        if board is None:
            return
        new_board1 = self.convert_board_to_strategy_vectors(board)
        new_board2 = self.convert_board_representation_negative_positive(board)
        new_board2 = np.append(new_board2, np.zeros((ROWS, 1)), axis=1)
        return np.append(new_board1, new_board2, axis=0)

    def convert_board_to_strategy_vectors_depth2_with_feature1(self, board):
        if board is None:
            return
        if self.id == 1:
            other_player_id = 2
        else:
            other_player_id = 1
        new_board = np.zeros((4, COLS + 1))
        legal_actions_first = get_legal_moves(board)
        for col in legal_actions_first:
            check_for_win_mine = check_for_win(make_move(board, col, self.id), self.id, col)
            if check_for_win_mine:
                new_board[0][col] = 1
                new_board[0][COLS] = 1
            check_for_win_other = check_for_win(make_move(board, col, other_player_id), other_player_id, col)
            if check_for_win_other:
                new_board[1][col] = 1
                new_board[1][COLS] = 1

        if not (1 in new_board[0]) and not (1 in new_board[1]):
            if not len(legal_actions_first) == 0:
                random_action_mine = np.random.choice(legal_actions_first)
                board_after_my_step = make_move(board, random_action_mine, self.id)
                legal_actions_second = get_legal_moves(board_after_my_step)
                if not len(legal_actions_second) == 0:
                    random_action_other = np.random.choice(legal_actions_second)
                    board_after_other_step = make_move(board_after_my_step, random_action_other, other_player_id)
                    legal_actions_second_other = get_legal_moves(board_after_other_step)

                    for col in legal_actions_second_other:
                        if check_for_win(make_move(board_after_other_step, col, self.id), self.id, col):
                            new_board[2][col] = 1
                            new_board[2][COLS] = 1
                        if check_for_win(make_move(board_after_other_step, col, other_player_id), other_player_id, col):
                            new_board[3][col] = 1
                            new_board[3][COLS] = 1

        return new_board

    def convert_board_to_strategy_vectors_depth2_with_feature2(self, board):
        if board is None:
            return
        if self.id == 1:
            other_player_id = 2
        else:
            other_player_id = 1
        new_board = np.zeros((4, COLS + 1))
        legal_actions_first = get_legal_moves(board)
        for col in legal_actions_first:
            check_for_win_mine = check_for_win(make_move(board, col, self.id), self.id, col)
            if check_for_win_mine:
                new_board[0][col] = 1
            check_for_win_other = check_for_win(make_move(board, col, other_player_id), other_player_id, col)
            if check_for_win_other:
                new_board[1][col] = 1
            if board[ROWS-1,int(COLS/2)]==0:
                new_board[0][int(COLS/2)]= 1
                new_board[0][COLS] = 1

        if not (1 in new_board[0]) and not (1 in new_board[1]):
            if not len(legal_actions_first) == 0:
                random_action_mine = np.random.choice(legal_actions_first)
                board_after_my_step = make_move(board, random_action_mine, self.id)
                legal_actions_second = get_legal_moves(board_after_my_step)
                if not len(legal_actions_second) == 0:
                    random_action_other = np.random.choice(legal_actions_second)
                    board_after_other_step = make_move(board_after_my_step, random_action_other, other_player_id)
                    legal_actions_second_other = get_legal_moves(board_after_other_step)

                    for col in legal_actions_second_other:
                        if check_for_win(make_move(board_after_other_step, col, self.id), self.id, col):
                            new_board[2][col] = 1
                        if check_for_win(make_move(board_after_other_step, col, other_player_id), other_player_id, col):
                            new_board[3][col] = 1
                        if board_after_other_step[ROWS-1,int(COLS/2)]==0:
                            new_board[2][int(COLS/2)]= 1
                            new_board[2][COLS] = 1

        return new_board


class NeuralNetwork:
    def __init__(self, hidden_layers, input_shape, load_from=None, session=None, input_=None):
        """Create an ANN with fully connected hidden layers of width
        hidden_layers."""
        self.initialized = False
        self.load_from = load_from
        self.load_weights()
        self.session = tf.Session() if session is None else session
        if input_ is None:
            self.input = tf.placeholder(tf.float32, shape=(None, input_shape), name="input")
        else:
            self.input = input_

        # create layers
        self.layers = [self.input]

        for i, width in enumerate(hidden_layers):
            a = self.affine("hidden{}".format(i), self.layers[-1], width, index=i)
            self.layers.append(a)
        self.output = self.affine("output", self.layers[-1], COLS, relu=False, index=len(hidden_layers))
        self.probabilities = tf.nn.softmax(self.output, name="probabilities")

        self.output_max = tf.reduce_max(self.output, axis=1)

    def load_weights(self):
        self.weights = []
        self.biases = []
        try:
            load_from = str(Path(os.path.dirname(os.path.abspath(__file__))).parent) + "/models/" + self.load_from
            print(load_from)
            with open(load_from, 'rb') as archive:
                [self.weights, self.biases, epsilon] = pickle.load(archive)
            self.initialized = True
        except:
            return

    def weights_iter(self):
        """Iterate over all the weights of the network."""
        for w in self.weights:
            yield w

    def biases_iter(self):
        """Iterate over all the biases of the network."""
        for b in self.biases:
            yield b

    def affine(self, name_scope, input_tensor, out_channels, relu=True, residual=False, index=0):
        """Create a fully-connected affaine layer."""
        input_shape = input_tensor.get_shape().as_list()
        input_channels = input_shape[-1]
        with tf.variable_scope(name_scope):
            if not self.initialized:
                W = tf.get_variable("weights", initializer=tf.truncated_normal(
                    [input_channels, out_channels], stddev=1.0 / np.sqrt(float(input_channels))))
                b = tf.get_variable("biases", initializer=tf.zeros([out_channels]))
                self.weights.append(W)
                self.biases.append(b)
            else:
                W = tf.get_variable("weights", initializer=tf.constant(self.weights[index]))
                self.weights[index] = W
                b = tf.get_variable("biases", initializer=tf.constant(self.biases[index]))
                self.biases[index] = b

            A = tf.matmul(input_tensor, W) + b

            if relu:
                R = tf.nn.relu(A)
                if residual:
                    return R + input_tensor
                else:
                    return R
            else:
                return A

    def take(self, indices):
        """Return an operation that takes values from network outputs.
        e.g. NN.predict_max() == NN.take(NN.predict_argmax())
        """

        mask = tf.one_hot(indices=indices, depth=COLS, dtype=tf.bool, on_value=True, off_value=False, axis=-1)
        return tf.boolean_mask(self.output, mask)


class Database:
    def __init__(self, state_dim=(ROWS, COLS), db_size=DB_SIZE):
        self.state_dim = state_dim
        self.db_size = db_size
        self.num_of_items = 0
        self.full = False
        self.DB = np.rec.recarray(self.db_size, dtype=[
            ("s1", np.int32, self.state_dim),
            ("s2", np.int32, self.state_dim),
            ("a", np.int32),
            ("r", np.float32),
            ('done', np.bool)])

    def add_item(self, s1, s2, a, r, done):
        if not s1 is None and a:
            self.DB['s1'][self.num_of_items] = s1
            self.DB['s2'][self.num_of_items] = s2
            self.DB['a'][self.num_of_items] = a
            self.DB['r'][self.num_of_items] = r
            self.DB['done'][self.num_of_items] = done
            self.num_of_items += 1
            # if db is full insert next item at the beginning
            if self.num_of_items + 1 > self.db_size:
                self.num_of_items = 0

    def take_sample(self, from_idx, batch_size=BATCH_SIZE):
        if from_idx + batch_size > self.num_of_items:
            ret_batch = self.DB[from_idx:self.num_of_items]
            ret_size = self.num_of_items - from_idx
            p = np.random.permutation(self.num_of_items)
            self.DB[:self.num_of_items] = np.rec.array(self.DB[:self.num_of_items][p])
            return ret_batch, ret_size
        return self.DB[from_idx:from_idx + batch_size], batch_size


def get_legal_moves(board_state):
    """
    Given a board state, the function returns the legal moves on that board.
    :param board_state: the board state received from the game.
    :return: array of legal moves.
    """

    legal_cols = np.array(np.where(board_state[0, :] == EMPTY_VAL))
    return np.reshape(legal_cols, (legal_cols.size,))


def check_for_win(board, player_id, col):
    """
    check the board to see if last move was a winning move.
    :param board: the new board
    :param player_id: the player who made the move
    :param col: his action
    :return: True iff the player won with his move
    """

    row = 0

    # check which row was inserted last:
    for i in range(ROWS):
        if board[ROWS - 1 - i, col] == EMPTY_VAL:
            row = ROWS - i
            break

    # check horizontal:
    vec = board[row, :] == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True

    # check vertical:
    vec = board[:, col] == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True

    # check diagonals:
    vec = np.diagonal(board, col - row) == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True
    vec = np.diagonal(np.fliplr(board), COLS - col - 1 - row) == player_id
    if np.any(np.convolve(WIN_MASK, vec, mode="valid") == 4):
        return True

    return False


def make_move(board, action, player_id):
    """
    return a new board with after performing the given move.
    :param board: original board
    :param action: move to make (column)
    :param player_id: player that made the move
    :return: new board after move was made
    """
    row = np.max(np.where(board[:, action] == EMPTY_VAL))
    new_board = np.copy(board)
    new_board[row, action] = player_id

    return new_board


def done(state, reward):
    return get_legal_moves(state).shape[0] == 0 or reward!=0
