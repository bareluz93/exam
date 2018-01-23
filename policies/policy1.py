from policies import base_policy as bp
import numpy as np
import tensorflow as tf
import pickle
from pathlib import Path
import os.path
import copy

GAMMA = 0.999

BATCH_SIZE = 100
NUM_OF_BATCHES = 1000

LEARNING_RATE = 0.001
EPSILON = 1

DB_SIZE = 10000
INPUT = "strategy"

EMPTY_VAL = 0
PLAYER1_ID = 1
PLAYER2_ID = 2

ROWS = 6
COLS = 7
WIN_MASK = np.ones(4)
ACTIONS = [0, 1, 2, 3, 4, 5, 6]


class Policy1(bp.Policy):  # todo change documentation
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
        policy_args['epsilon'] = float(
            policy_args['epsilon']) if 'epsilon' in policy_args else EPSILON
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
        self.session = tf.Session()
        if INPUT=="strategy":
            self.input_shape=(2,COLS)
            self.input_flat_shape=2*COLS
        elif INPUT =="hot":
            self.input_shape=(2,ROWS,COLS)
            self.input_flat_shape=2*ROWS*COLS
        elif INPUT =="negative_positive" or INPUT =="original":
            self.input_shape=(ROWS,COLS)
            self.input_flat_shape=ROWS*COLS
        self.nn = NeuralNetwork(self.hidden_layers,self.input_flat_shape ,session=self.session, load_from=self.load_from)

        # initialize data base
        self.db = Database(state_dim=self.input_shape)

        # initialize learning parameters
        self.actions = tf.placeholder(tf.int32, (None,), "actions")
        self.q_values = self.nn.take(self.actions)
        self.q_estimation = tf.placeholder(tf.float32, (None,), name="q_estimation")
        self.loss = tf.reduce_mean((self.q_estimation - self.q_values) ** 2)
        # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = tf.train.AdamOptimizer()
        self.train_op = self.optimizer.minimize(self.loss)
        self.session.run(tf.global_variables_initializer())

    def learn(self, round, prev_state, prev_action, reward, new_state, too_slow):
        self.log('learn round ' + str(round))

        # update database
        converted_prev_state,converted_new_state=self.convert_boards_srategy(prev_state,new_state)
        self.db.add_item(converted_prev_state, converted_new_state, prev_action, reward, done(new_state, reward))
        # self.log("real previous state")
        # self.log("\n" + str(prev_state))
        # self.log("converted_prev:")
        # self.log("\n" + str(converted_prev_state))
        # self.log("converted_new:")
        # self.log("\n" + str(converted_new_state))
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
            inputs = (batch.s1).reshape(shape[0], shape[1] * shape[2])#todo check dimensions for one hot
            second_states = second_states.reshape(shape[0], shape[1] * shape[2])#todo check dimensions for one hot

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
        self.log("\nepsilon: "+str(self.epsilon)+"\n")
        self.epsilon*=1-round/self.game_duration

    def act(self, round, prev_state, prev_action, reward, new_state, too_slow):  # TODO add epsilon greedy
        self.log('act round ' + str(round))

        # update database
        converted_prev_state,converted_new_state=self.convert_boards_srategy(prev_state,new_state)
        # self.log("real previous state")
        # self.log("\n" + str(prev_state))
        # self.log("converted_prev:")
        # self.log("\n" + str(converted_prev_state))
        self.db.add_item(converted_prev_state, converted_new_state, prev_action, reward, done(new_state, reward))

        legal_actions = get_legal_moves(new_state)
        if np.random.uniform()<self.epsilon:
            action= np.random.choice(legal_actions)
        else:
            # out = self.nn.session.run(self.nn.output, feed_dict={self.nn.input: new_state.reshape(1, ROWS * COLS)})[0]
            out = self.nn.session.run(self.nn.output, feed_dict={self.nn.input: converted_new_state.reshape(1, self.input_flat_shape)})[0]
            legal_tup = np.asarray([(i, out[i]) for i in legal_actions])
            idx = np.argmax(legal_tup[:, 1])
            action = int(legal_tup[idx][0])
        # self.log('out: '+str(out_probs))
        # self.log("legal actions"+str(legal_actions))
        # self.log('action chosen: ' + str(action))

        return action

    def save_model(self):
        return [[self.nn.session.run(w) for w in self.nn.weights_iter()],
                [self.nn.session.run(b) for b in self.nn.biases_iter()]], self.save_to

    def convert_boards_srategy(self,prev_state,new_state):

        if INPUT=="strategy":
            converted_prev_state=convert_board_to_strategy_vectors(prev_state,self.id)
            converted_new_state=convert_board_to_strategy_vectors(new_state,self.id)
        elif INPUT =="hot":
            converted_prev_state=convert_board_to_one_hot(prev_state,self.id)
            converted_new_state=convert_board_to_one_hot(new_state,self.id)
        elif INPUT =="negative_positive":
            converted_prev_state=convert_board_representation_negative_positive(prev_state,self.id)
            converted_new_state=convert_board_representation_negative_positive(new_state,self.id)
        elif INPUT == "original":
            converted_prev_state=prev_state
            converted_new_state=new_state
        return converted_prev_state,converted_new_state


class NeuralNetwork:
    def __init__(self, hidden_layers, input_shape, load_from=None, session=None, input_=None):
        """Create an ANN with fully connected hidden layers of width
        hidden_layers."""
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
            if self.num_of_items+1 > self.db_size:
                self.num_of_items = 0

    def take_sample(self, from_idx, batch_size=BATCH_SIZE):
        if from_idx + batch_size > self.num_of_items:
            ret_batch=self.DB[from_idx:]
            ret_size=self.num_of_items - from_idx
            p = np.random.permutation(self.num_of_items)
            self.DB[:self.num_of_items] = np.rec.array(self.DB[:self.num_of_items][p])
            return ret_batch,ret_size
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


def convert_board_representation_negative_positive(board, player_id):
    if board is None:
        return
    if player_id == 1:
        other_player_id = 2
    else:
        other_player_id = 1
    new_board = copy.deepcopy(board)
    new_board[new_board == player_id] = 1
    new_board[new_board == other_player_id] = -1
    return new_board


def convert_board_to_one_hot(board, player_id):
    if board is None:
        return
    if player_id == 1:
        other_player_id = 2
    else:
        other_player_id = 1
    new_board = np.zeros(2, board.shape[0], board.shape[1])
    new_board[0][board == player_id] = 1
    new_board[1][board == other_player_id] = 1
    return new_board


def convert_board_to_strategy_vectors(board, player_id):
    if board is None:
        return
    if player_id == 1:
        other_player_id = 2
    else:
        other_player_id = 1
    new_board = np.zeros((2, COLS))
    legal_actions=get_legal_moves(board)
    for col in legal_actions:
        if check_for_win(make_move(board, col, player_id), player_id, col):
            new_board[0][col] = 1
        if check_for_win(make_move(board, col, other_player_id), other_player_id, col):
            new_board[1][col] = 1
    return new_board

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