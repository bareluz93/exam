from policies import base_policy as bp
import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path
import os.path

GAMMA = 0.999

BATCH_SIZE = 100
NUM_OF_BATCHES = 1000

LEARNING_RATE = 0.001

DB_SIZE = 10000

EMPTY_VAL = 0
PLAYER1_ID = 1
PLAYER2_ID = 2

ROWS = 6
COLS = 7
WIN_MASK = np.ones(4)
ACTIONS = [0, 1, 2, 3, 4, 5, 6]


class Policy2(bp.Policy):  # todo change documentation
    """
    An agent performing the Minmax Algorithm for a given depth. The agent will
    return the right moves if there is a win or a correct defensive move for
    the given depth, and otherwise act randomly.
    """

    def cast_string_args(self, policy_args):  # todo change
        policy_args['save_to'] = str(policy_args['save_to']) if 'save_to' in policy_args else 'policy1.model.pkl'
        policy_args['load_from'] = str(policy_args['load_from']) if 'load_from' in policy_args else None
        policy_args['gamma'] = float(policy_args['gamma']) if 'gamma' in policy_args else GAMMA
        policy_args['batch_size'] = int(policy_args['batch_size']) if 'batch_size' in policy_args else BATCH_SIZE
        policy_args['num_of_batches'] = int(
            policy_args['num_of_batches']) if 'num_of_batches' in policy_args else NUM_OF_BATCHES
        policy_args['learning_rate'] = float(
            policy_args['learning_rate']) if 'learning_rate' in policy_args else LEARNING_RATE
        return policy_args

    def init_run(self):  # TODO READ FROM DB
        # load model
        try:
            model = pickle.load(open(self.load_from, 'rb'))
            self.log("Model found", 'STATUS')
            self.W = [tf.Variable(tf.constant(w)) for w in model[0]]
            self.b = [tf.Variable(tf.constant(b)) for b in model[1]]

        except:
            self.log("Model not found, initializing random weights.", 'STATUS')

        # initialize neural network
        self.hidden_layers = [50, 50, 50]
        # self.hidden_layers = [100, 100, 100]rce
        # self.hidden_layers = [50]
        self.session = tf.Session()
        self.nn = NeuralNetwork(hidden_layers=self.hidden_layers, session=self.session, load_from=self.load_from)

        # initialize data base
        self.db = Database()

        # initialize learning parameters
        self.actions = tf.placeholder(tf.int32, (None,), "actions")
        self.q_values = self.nn.take(self.actions)
        self.q_estimation = tf.placeholder(tf.float32, (None,), name="q_estimation")
        self.loss = tf.reduce_mean((self.q_estimation - self.q_values) ** 2)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        self.log('learn round ' + str(round))
        # update database
        self.db.add_item(prev_state, new_state, prev_action, reward, done(new_state, reward))

        # self.log('database:.......................')
        # self.log(self.db.DB[:10])

        # train in batches
        from_idx = 0
        for i in range(self.num_of_batches):
            batch, real_batch_size = self.db.take_sample(from_idx, self.batch_size)
            # self.log('batch:............... ')
            # self.log(batch)
            second_states = batch.s2
            shape = second_states.shape
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

            from_idx += self.batch_size
            if from_idx > self.db.num_of_items:
                break

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):  # TODO add epsilon greedy
        self.log('act round ' + str(round))

        # update database
        self.db.add_item(prev_state, new_state, prev_action, reward, done(new_state, reward))

        legal_actions = get_legal_moves(new_state)

        out = self.nn.session.run(self.nn.output, feed_dict={self.nn.input: new_state.reshape(1, ROWS * COLS)})[0]
        legal_tup = np.asarray([(i, out[i]) for i in legal_actions])
        idx = np.argmax(legal_tup[:, 1])
        action = int(legal_tup[idx][0])
        # self.log('out: '+str(out_probs))
        # self.log("legal actions"+str(legal_actions))
        self.log('action chosen: ' + str(action))
        return action
        # legal_actions = get_legal_moves(new_state)
        # return np.random.choice(legal_actions)

    def save_model(self):
        return [[self.nn.session.run(w) for w in self.nn.weights_iter()],
                [self.nn.session.run(b) for b in self.nn.biases_iter()]], self.save_to


class NeuralNetwork:
    def __init__(self, hidden_layers, load_from=None, session=None, input_=None):
        """Create an ANN with fully connected hidden layers of width
        hidden_layers."""
        self.load_from = load_from
        self.load_weights()

        self.session = tf.Session() if session is None else session
        if input_ is None:
            self.input = tf.placeholder(tf.float32, shape=(None, ROWS * COLS), name="input")
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
        # self.output_argmax = tf.argmax(self.output, axis=1)

    def load_weights(self):
        self.weights = []
        self.biases = []
        if self.load_from:
            # print(self.load_from)
            # print(Path(self.load_from).parent)
            load_from = str(Path(os.path.dirname(os.path.abspath(__file__))).parent) + "/models/" + self.load_from
            print(load_from)
            with open(load_from, 'rb') as archive:
                [self.weights, self.biases] = pickle.load(archive)
                print(self.weights)
                print("-------------------------")
                print(self.biases)
                print("---------------------------------------------------------------\n\n\n\n\n")
                print(len(self.weights), len(self.biases))

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
            if not self.load_from:
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

    # def predict(self, op, inputs_feed, batch_size=None):  # TODO flatten input
    #     feed_dict = {self.input: inputs_feed}
    #     return self.run_op_in_batches(self.session, op, feed_dict, batch_size)
    #
    # def predict_exploration(self, inputs_feed, epsilon=0.1, batch_size=None):
    #     """Return argmax with probability (1-epsilon), and random value with
    #     probabilty epsilon."""
    #
    #     n = len(inputs_feed)
    #     # out = self.predict_argmax(inputs_feed, batch_size)
    #     out = self.predict(self.output_argmax, inputs_feed, batch_size)
    #     exploration = np.random.random(n) < epsilon
    #     out[exploration] = np.random.choice(COLS, exploration.sum())
    #
    #     return out

    def train_in_batches(self, train_op, feed_dict, n_batches, batch_size, balanced=False):
        """Train the network by randomly sub-sampling feed_dict."""

        keys = tuple(feed_dict.keys())
        ds = [feed_dict[k] for k in keys]
        for i in range(n_batches):
            batch = self.next_batch(batch_size, *ds)
            d = {k: b for (k, b) in zip(keys, batch)}
            self.session.run(train_op, d)

    def accuracy(self, accuracy_op, feed_dict, batch_size):
        """Return the average value of an accuracy op by running the network
        on small batches of feed_dict."""

        return self.run_op_in_batches(self.session, accuracy_op,
                                      feed_dict, batch_size).mean()

        # def next_batch(self, batch_size, *args):
        #     X = [a.copy() for a in args]
        #     n = X[0].shape[0]
        #     if batch_size > n:
        #         batch_size = n
        #     if self.batch_idx + batch_size > n:
        #         self.batch_idx = 0
        #     batch = X[self.batch_idx: self.batch_idx + batch_size]
        #     self.batch_idx += batch_size
        #     return tuple(a[batch] for a in X)


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
            return self.DB[from_idx:], self.num_of_items - from_idx
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


def done(state, reward):
    return get_legal_moves(state).shape[0] == 0 or reward
