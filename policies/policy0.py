from policies import base_policy as bp
import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path  # todo check if needed

GAMMA = 0.999

BATCH_SIZE = 1000

LEARNING_RATE = 0.001

DB_SIZE = 10000000

EMPTY_VAL = 0
PLAYER1_ID = 1
PLAYER2_ID = 2

ROWS = 6
COLS = 7
WIN_MASK = np.ones(4)
ACTIONS = [0, 1, 2, 3, 4, 5, 6]


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
    return get_legal_moves(state).shape[0]==0 or reward


class Policy0(bp.Policy):  # todo change documentation
    """
    An agent performing the Minmax Algorithm for a given depth. The agent will
    return the right moves if there is a win or a correct defensive move for
    the given depth, and otherwise act randomly.
    """

    def cast_string_args(self, policy_args):  # todo change

        policy_args['save_to'] = str(policy_args['save_to']) if 'save_to' in policy_args else 'policy0.model.pkl'
        return policy_args

    def init_run(self):  # TODO READ FROM DB

        self.db = History((ROWS, COLS), DB_SIZE)
        self.hidden_layers = [50, 50, 50]

        self.session = tf.Session()
        self.nn = NeuralNetwork(self.hidden_layers, self.session)
        self.session.run(tf.global_variables_initializer())

        self.actions = tf.placeholder(tf.int32, (None,), "actions")  # TODO CHECK DIMENSIONS
        self.q_values = self.nn.take(self.actions)
        self.q_estimation = tf.placeholder(tf.float32, (None,), name="q_estimation")
        self.loss = tf.reduce_mean((self.q_estimation - self.q_values) ** 2)

        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        self.train_op = self.optimizer.minimize(self.loss)

        try:
            model = pickle.load(open(self.load_from, 'rb'))
            self.nn.load_weights(model)

        except:

            self.log("Model not found.", 'STATUS')
        return

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        self.db.store(prev_state, new_state, prev_action, reward)
        self.db.update_rewards(self.nn.output_max, self.nn.session, self.nn.input)
        v = self.nn.predict(self.nn.output_max, self.db.DB.s2, BATCH_SIZE)
        q = self.db.DB.r + (~self.done(new_state, reward) * GAMMA * v)

        feed_dict = {
            self.nn.input: self.db.DB.s1,
            self.actions: self.db.DB.a,
            self.q_estimation: q
        }

        self.nn.train_in_batches(self.train_op, feed_dict, self.args.num_batches, BATCH_SIZE)

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):
        self.db.store(prev_state, new_state, prev_action, reward)
        legal_actions = get_legal_moves(new_state)
        action = self.nn.session.run(self.nn.output_argmax,
                                     feed_dict={self.nn.input: new_state.reshape(1, ROWS * COLS)})[0]
        if action in legal_actions:
            return action
        else:
            return np.random.choice(legal_actions)

    def save_model(self):
        return [self.nn.session.run(var) for var in self.nn.vars()], None


class NeuralNetwork:
    @staticmethod
    def run_op_in_batches(session, op, batch_dict={}, batch_size=None,
                          extra_dict={}):

        """Return the result of op by running the network on small batches of
        batch_dict."""

        if batch_size is None:
            return session.run(op, feed_dict={**batch_dict, **extra_dict})

        # Probably the least readable form to get an arbitrary item from a dict
        n = len(next(iter(batch_dict.values())))

        s = []
        for i in range(0, n, batch_size):
            bd = {k: b[i: i + batch_size] for (k, b) in batch_dict.items()}
            s.append(session.run(op, feed_dict={**bd, **extra_dict}))

        if s[0] is not None:
            if np.ndim(s[0]):
                return np.concatenate(s)
            else:
                return np.asarray(s)

    def __init__(self, hidden_layers, session=None,
                 name_prefix="", input_=None):
        """Create an ANN with fully connected hidden layers of width
        hidden_layers."""

        self.saver = None

        self.name_prefix = name_prefix
        self.weights = []
        self.biases = []

        self.session = tf.Session() if session is None else session
        if input_ is None:
            self.input = tf.placeholder(tf.float32,
                                        shape=(None, ROWS * COLS),
                                        name="{}input".format(self.name_prefix))
        else:
            self.input = input_
        self.layers = [self.input]

        for i, width in enumerate(hidden_layers):
            a = self.affine("{}hidden{}".format(self.name_prefix, i),
                            self.layers[-1], width)
            self.layers.append(a)

        self.output = self.affine("{}output".format(self.name_prefix), self.layers[-1], COLS, relu=False)
        self.probabilities = tf.nn.softmax(self.output, name="{}probabilities".format(self.name_prefix))
        self.output_max = tf.reduce_max(self.output, axis=1)
        self.output_argmax = tf.argmax(self.output, axis=1)
        self.batch_idx = 0

    def load_weights(self, model):
        self.weights = model[:len(self.weights)]
        self.biases = model[len(self.weights):]

    def vars(self):
        """Iterate over all the variables of the network."""

        for w in self.weights:
            yield w
        for b in self.biases:
            yield b

    def affine(self, name_scope, input_tensor, out_channels, relu=True, residual=False):
        """Create a fully-connected affaine layer."""
        # tf.reshape(input_tensor,[None,input_tensor.get_shape().as_list()[0]*input_tensor.get_shape().as_list()[1]])
        input_shape = input_tensor.get_shape().as_list()
        input_channels = input_shape[-1]
        with tf.variable_scope(name_scope):
            W = tf.get_variable("weights", initializer=tf.truncated_normal(
                [input_channels, out_channels], stddev=1.0 / np.sqrt(float(input_channels))))
            b = tf.get_variable("biases", initializer=tf.zeros([out_channels]))

            self.weights.append(W)
            self.biases.append(b)

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

    def predict(self, op, inputs_feed, batch_size=None):  # TODO flatten input
        feed_dict = {self.input: inputs_feed}
        return self.run_op_in_batches(self.session, op, feed_dict, batch_size)

    def predict_exploration(self, inputs_feed, epsilon=0.1, batch_size=None):
        """Return argmax with probability (1-epsilon), and random value with
        probabilty epsilon."""

        n = len(inputs_feed)
        # out = self.predict_argmax(inputs_feed, batch_size)
        out = self.predict(self.output_argmax, inputs_feed, batch_size)
        exploration = np.random.random(n) < epsilon
        out[exploration] = np.random.choice(COLS, exploration.sum())

        return out

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

    def next_batch(self, batch_size, *args):
        X = [a.copy() for a in args]
        n = X[0].shape[0]
        if batch_size > n:
            batch_size = n
        if self.batch_idx + batch_size > n:
            self.batch_idx = 0
        batch = X[self.batch_idx: self.batch_idx + batch_size]
        self.batch_idx += batch_size
        return tuple(a[batch] for a in X)


class History:
    """Holds previous games and allows sampling random combinations of
        (state, action, new state, reward)
    """

    def __init__(self, state_dim, db_size):
        """Create new DB of size db_size."""

        self.state_dim = state_dim
        self.db_size = db_size

        self.DB = np.rec.recarray(self.db_size, dtype=[
            ("s1", np.int32, self.state_dim),
            ("s2", np.int32, self.state_dim),
            ("a", np.int32),
            ("r", np.float32),
        ])
        self.clear()

    def clear(self):
        """Remove all entries from the DB."""

        self.index = 0
        self.n_items = 0
        self.full = False
        self.DB = np.rec.recarray(self.db_size, dtype=[
            ("s1", np.int32, self.state_dim),
            ("s2", np.int32, self.state_dim),
            ("a", np.int32),
            ("r", np.float32),
        ])

    def update_rewards(self, output_max, sess, nn_input):
        for item in self.DB:

            v = sess.run(output_max, feed_dict={nn_input: item.s2.reshape(1, ROWS * COLS)}).reshape(1)
            reward = item.r
            if not done(item.s1, item.r):
                reward += GAMMA * v
            item.r = reward

    def store(self, s1, s2, a, r):
        """Store new samples in the DB."""
        if not s1 is None and a:
            self.DB['s1'][self.index] = s1
            self.DB['s2'][self.index] = s2
            self.DB['a'][self.index] = a
            self.DB['r'][self.index] = r
            self.index += 1
            if self.index > self.db_size:
                self.full = True
                self.index = 0

    def iter_samples(self, sample_size, n_samples):
        """Iterate over random samples from the DB."""

        if sample_size == 0:
            sample_size = self.n_items

        ind = self.n_items
        for i in range(n_samples):
            end = ind + sample_size
            if end > self.n_items:
                ind = 0
                end = sample_size
                p = np.random.permutation(self.n_items)
                db = np.rec.array(self.DB[p])
            yield db[ind: end]
            ind = end
