import logging
from collections import namedtuple
import tensorflow as tf
from tensorflow.contrib import metrics, slim
from tensorflow.contrib.metrics import streaming_mean
from networks.meanTeacher.network import resnet_1, fully_connected
from networks.meanTeacher.framework.framework import ema_variable_scope, name_variable_scope, assert_shape, HyperparamVariables
from networks.meanTeacher.framework.string_utils import *
logging.basicConfig(level=logging.INFO, filename= 'train_log',filemode='a')
LOG = logging.getLogger('train_log')
LOG.setLevel(logging.INFO)


class Model:
    DEFALT_HYPERPARAMS = {
        # Consistency hyperparameters
        'output_num': 2,
        'ema_consistency': True,
        'apply_consistency_to_labeled': True,
        'max_consistency_cost': 100.0,
        'ema_decay_during_rampup': 0.99,
        'ema_decay_after_rampup': 0.999,
        'consistency_trust': 0.0,
        'num_logits': 1,  # Either 1 or 2
        'logit_distance_cost': 0.0,  # Matters only with 2 outputs

        # Optimizer hyperparameters
        'max_learning_rate': 0.001,
        'adam_beta_1_before_rampdown': 0.9,
        'adam_beta_1_after_rampdown': 0.5,
        'adam_beta_2_during_rampup': 0.99,
        'adam_beta_2_after_rampup': 0.999,
        'adam_epsilon': 1e-8,

        # Architecture hyperparameters
        'input_noise': 0.01,
        'student_dropout_probability': 0.5,
        'teacher_dropout_probability': 0.5,

        # Training schedule
        'rampup_length': 40000,
        'rampdown_length': 25000,
        'training_length': 150000,

        # Input augmentation
        'flip_horizontally': False,
        'translate': True,

        # Whether to scale each input image to mean=0 and std=1 per channel
        # Use False if input is already normalized in some other way
        'normalize_input': True,

        # Output schedule
        'print_span': 20,
        'evaluation_span': 100,
    }

    def __init__(self, checkpoint_path, tensorboard_path, input_shape):

        self.checkpoint_path = checkpoint_path
        self.tensorboard_path = tensorboard_path
        with tf.name_scope('placeholders'):
            self.features = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape, name='features')
            self.labels = tf.placeholder(dtype=tf.int32, shape=(None,), name='labels')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        tf.add_to_collection("init_in_init", self.global_step)
        self.hyper = HyperparamVariables(self.DEFALT_HYPERPARAMS)
        for var in self.hyper.variables.values():
            tf.add_to_collection("init_in_init", var)

        with tf.name_scope("ramps"):
            sigmoid_rampup_value = sigmoid_rampup(self.global_step, self.hyper['rampup_length'])
            sigmoid_rampdown_value = sigmoid_rampdown(self.global_step,
                                                      self.hyper['rampdown_length'],
                                                      self.hyper['training_length'])
            self.learning_rate = tf.multiply(sigmoid_rampup_value * sigmoid_rampdown_value,
                                             self.hyper['max_learning_rate'],
                                             name='learning_rate')
            self.adam_beta_1 = tf.add(sigmoid_rampdown_value * self.hyper['adam_beta_1_before_rampdown'],
                                      (1 - sigmoid_rampdown_value) * self.hyper['adam_beta_1_after_rampdown'],
                                      name='adam_beta_1')
            self.cons_coefficient = tf.multiply(sigmoid_rampup_value,
                                                self.hyper['max_consistency_cost'],
                                                name='consistency_coefficient')
            step_rampup_value = step_rampup(self.global_step, self.hyper['rampup_length'])
            self.adam_beta_2 = tf.add((1 - step_rampup_value) * self.hyper['adam_beta_2_during_rampup'],
                                      step_rampup_value * self.hyper['adam_beta_2_after_rampup'],
                                      name='adam_beta_2')
            self.ema_decay = tf.add((1 - step_rampup_value) * self.hyper['ema_decay_during_rampup'],
                                    step_rampup_value * self.hyper['ema_decay_after_rampup'],
                                    name='ema_decay')
        (
            (self.class_logits_1, self.cons_logits_1),
            (self.class_logits_2, self.cons_logits_2),
            (self.class_logits_ema, self.cons_logits_ema)
        ) = inference(
            self.features,
            is_training=self.is_training,
            ema_decay=self.ema_decay,
            input_noise=self.hyper['input_noise'],
            student_dropout_probability=self.hyper['student_dropout_probability'],
            teacher_dropout_probability=self.hyper['teacher_dropout_probability'],
            normalize_input=self.hyper['normalize_input'],
            flip_horizontally=self.hyper['flip_horizontally'],
            translate=self.hyper['translate'],
            num_logits=self.hyper['num_logits'])

        self.output = tf.multiply(self.class_logits_1,1,name='output')
        with tf.name_scope("objectives"):
            self.mean_error_1, self.errors_1 = errors(self.class_logits_1, self.labels)
            self.mean_error_ema, self.errors_ema = errors(self.class_logits_ema, self.labels)

            self.mean_class_cost_1, self.class_costs_1 = classification_costs(
                self.class_logits_1, self.labels)
            self.mean_class_cost_ema, self.class_costs_ema = classification_costs(
                self.class_logits_ema, self.labels)

            labeled_consistency = self.hyper['apply_consistency_to_labeled']
            consistency_mask = tf.logical_or(tf.equal(self.labels, -1), labeled_consistency)
            self.mean_cons_cost_pi, self.cons_costs_pi = consistency_costs(
                self.cons_logits_1, self.class_logits_2, self.cons_coefficient, consistency_mask,
                self.hyper['consistency_trust'])
            self.mean_cons_cost_mt, self.cons_costs_mt = consistency_costs(
                self.cons_logits_1, self.class_logits_ema, self.cons_coefficient, consistency_mask,
                self.hyper['consistency_trust'])

            def l2_norms(matrix):
                l2s = tf.reduce_sum(matrix ** 2, axis=1)
                mean_l2 = tf.reduce_mean(l2s)
                return mean_l2, l2s

            self.mean_res_l2_1, self.res_l2s_1 = l2_norms(self.class_logits_1 - self.cons_logits_1)
            self.mean_res_l2_ema, self.res_l2s_ema = l2_norms(self.class_logits_ema - self.cons_logits_ema)
            self.res_costs_1 = self.hyper['logit_distance_cost'] * self.res_l2s_1
            self.mean_res_cost_1 = tf.reduce_mean(self.res_costs_1)
            self.res_costs_ema = self.hyper['logit_distance_cost'] * self.res_l2s_ema
            self.mean_res_cost_ema = tf.reduce_mean(self.res_costs_ema)

            self.mean_total_cost_pi, self.total_costs_pi = total_costs(
                self.class_costs_1, self.cons_costs_pi, self.res_costs_1)
            self.mean_total_cost_mt, self.total_costs_mt = total_costs(
                self.class_costs_1, self.cons_costs_mt, self.res_costs_1)
            assert_shape(self.total_costs_pi, [3])
            assert_shape(self.total_costs_mt, [3])
            self.cost_to_be_minimized = tf.cond(self.hyper['ema_consistency'],
                                                lambda: self.mean_total_cost_mt,
                                                lambda: self.mean_total_cost_pi)

        with tf.name_scope("train_step"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step_op = adam_optimizer(self.cost_to_be_minimized,
                                                       self.global_step,
                                                       learning_rate=self.learning_rate,
                                                       beta1=self.adam_beta_1,
                                                       beta2=self.adam_beta_2,
                                                       epsilon=self.hyper['adam_epsilon'])

        self.training_control = training_control(self.global_step,
                                                 self.hyper['print_span'],
                                                 self.hyper['evaluation_span'],
                                                 self.hyper['training_length'])
        self.training_metrics = {
            "learning_rate": self.learning_rate,
            "adam_beta_1": self.adam_beta_1,
            "adam_beta_2": self.adam_beta_2,
            "ema_decay": self.ema_decay,
            "cons_coefficient": self.cons_coefficient,
            "train/error/1": self.mean_error_1,
            "train/error/ema": self.mean_error_ema,
            "train/class_cost/1": self.mean_class_cost_1,
            "train/class_cost/ema": self.mean_class_cost_ema,
            "train/cons_cost/pi": self.mean_cons_cost_pi,
            "train/cons_cost/mt": self.mean_cons_cost_mt,
            "train/res_cost/1": self.mean_res_cost_1,
            "train/res_cost/ema": self.mean_res_cost_ema,
            "train/total_cost/pi": self.mean_total_cost_pi,
            "train/total_cost/mt": self.mean_total_cost_mt,
        }
        '''monitor values on tensorboard'''
        for k,v in self.training_metrics.items():
            tf.summary.scalar(k,v)

        with tf.variable_scope("validation_metrics") as metrics_scope:
            self.metric_values, self.metric_update_ops = metrics.aggregate_metric_map({
                "eval/error/1": streaming_mean(self.errors_1),
                "eval/error/ema": streaming_mean(self.errors_ema),
                "eval/class_cost/1": streaming_mean(self.class_costs_1),
                "eval/class_cost/ema": streaming_mean(self.class_costs_ema),
                "eval/res_cost/1": streaming_mean(self.res_costs_1),
                "eval/res_cost/ema": streaming_mean(self.res_costs_ema),
            })
            metric_variables = slim.get_local_variables(scope=metrics_scope.name)
            self.metric_init_op = tf.variables_initializer(metric_variables)

        self.result_formatter = DictFormatter(
            order=["eval/error/ema", "error/1", "class_cost/1", "cons_cost/mt"],
            default_format='{name}: {value:>10.6f}',
            separator=",  ")
        self.result_formatter.add_format('error', '{name}: {value:>6.1%}')

        with tf.name_scope("initializers"):
            init_init_variables = tf.get_collection("init_in_init")
            train_init_variables = [
                var for var in tf.global_variables() if var not in init_init_variables
            ]
            self.init_init_op = tf.variables_initializer(init_init_variables)
            self.train_init_op = tf.variables_initializer(train_init_variables)

        self.saver = tf.train.Saver(max_to_keep=11)
        self.session = tf.Session()
        self.run(self.init_init_op)

        total_parameters = 0
        for variable in tf.trainable_variables():
            print(variable)
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            print(variable_parameters)
            total_parameters += variable_parameters
            print(total_parameters)


    def __setitem__(self, key, value):
        self.hyper.assign(self.session, key, value)

    def __getitem__(self, key):
        return self.hyper.get(self.session, key)

    def train(self, training_batches, evaluation_batches_fn):
        self.best_loss = 1
        self.patience = 0
        self.run(self.train_init_op, self.feed_dict(next(training_batches)))
        LOG.info("Model variables initialized")
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()
        merged = tf.summary.merge_all()
        for batch in training_batches:
            merge, i, results, _ = self.run([merged,self.global_step, self.training_metrics, self.train_step_op],
                                  self.feed_dict(batch))
            self.writer.add_summary(merge, i)
            step_control = self.get_training_control()
            if step_control['time_to_print']:
                LOG.info("step %5d:   %s", step_control['step'], self.result_formatter.format_dict(results))
            if step_control['time_to_stop']:
                break
            if step_control['time_to_evaluate']:
                if_stop = self.evaluate(evaluation_batches_fn)
                self.save_checkpoint()
                if if_stop:
                    LOG.info('Stop Training')
                    break
        _ = self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()

    def evaluate(self, evaluation_batches_fn):
        self.run(self.metric_init_op)
        for batch in evaluation_batches_fn():
            self.run(self.metric_update_ops,
                     feed_dict=self.feed_dict(batch, is_training=False))
        step = self.run(self.global_step)
        results = self.run(self.metric_values)
        '''early stoping'''
        loss = results['eval/class_cost/1']
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience = 0
        else:
            self.patience += 1

        if self.patience == 10:
            stop_training = True
        else:
            stop_training = False

        LOG.info("step %5d:   %s", step, self.result_formatter.format_dict(results))

        return stop_training


    def get_training_control(self):
        return self.session.run(self.training_control)

    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def feed_dict(self, batch, is_training=True):
        return {
            self.features: batch['x'],
            self.labels: batch['y'],
            self.is_training: is_training
        }

    def save_checkpoint(self):
        path = self.saver.save(self.session, self.checkpoint_path, global_step=self.global_step)
        LOG.info("Saved checkpoint: %r", path)

    def save_model(self):
        inputs = {"features" : self.features}
        outputs = {"output" : self.output}
        tf.saved_model.simple_save(session=self.session, export_dir=self.checkpoint_path,
                                   inputs = inputs, outputs=outputs)

    def save_tensorboard_graph(self):
        self.writer = tf.summary.FileWriter(self.tensorboard_path)
        self.writer.add_graph(self.session.graph)
        return self.writer.get_logdir()

    def restore_checkpoint(self, number):
        self.saver.restore(self.session, self.checkpoint_path+'-%s'%str(number))

    def test(self,test_x):
        class_logits_1 = self.session.run(
            self.class_logits_1,
            feed_dict = {self.features: test_x, self.is_training: True},
        )
        return class_logits_1



# ----------------------------------------------------------------------------------------------------------------------
Hyperparam = namedtuple("Hyperparam", ['tensor', 'getter', 'setter'])

def adam_optimizer(cost, global_step,
                   learning_rate=0.001, beta1=0.9, beta2=0.99, epsilon=1e-8,
                   name=None):
    with tf.name_scope(name, "adam_optimizer") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=beta1,
                                           beta2=beta2,
                                           epsilon=epsilon)
        return optimizer.minimize(cost, global_step=global_step, name=scope)

def training_control(global_step, print_span, evaluation_span, max_step, name=None):
    with tf.name_scope(name, "training_control"):
        return {
            "step": global_step,
            "time_to_print": tf.equal(tf.mod(global_step, print_span), 0),
            "time_to_evaluate": tf.equal(tf.mod(global_step, evaluation_span), 0),
            "time_to_stop": tf.greater_equal(global_step, max_step),
        }

def step_rampup(global_step, rampup_length):
    result = tf.cond(global_step < rampup_length,
                     lambda: tf.constant(0.0),
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="step_rampup")


def sigmoid_rampup(global_step, rampup_length):
    global_step = tf.to_float(global_step)
    rampup_length = tf.to_float(rampup_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, global_step) / rampup_length
        return tf.exp(-5.0 * phase * phase)

    result = tf.cond(global_step < rampup_length, ramp, lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampup")


def sigmoid_rampdown(global_step, rampdown_length, training_length):
    global_step = tf.to_float(global_step)
    rampdown_length = tf.to_float(rampdown_length)
    training_length = tf.to_float(training_length)
    def ramp():
        phase = 1.0 - tf.maximum(0.0, training_length - global_step) / rampdown_length
        return tf.exp(-12.5 * phase * phase)

    result = tf.cond(global_step >= training_length - rampdown_length,
                     ramp,
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampdown")

# ----------------------------------------------------------------------------------------------------------------------

def inference(inputs, is_training, ema_decay, input_noise, student_dropout_probability, teacher_dropout_probability,
              normalize_input, flip_horizontally, translate, num_logits):
    tower_args = dict(inputs=inputs,
                      is_training=is_training,
                      input_noise=input_noise,
                      normalize_input=normalize_input,
                      num_logits=num_logits)

    with tf.variable_scope("initialization") as var_scope:
        _ = tower(**tower_args, dropout_probability=student_dropout_probability, is_initialization=True)
    with name_variable_scope("primary", var_scope, reuse=True) as (name_scope, _):
        class_logits_1, cons_logits_1 = tower(**tower_args, dropout_probability=student_dropout_probability, name=name_scope)
    with name_variable_scope("secondary", var_scope, reuse=True) as (name_scope, _):
        class_logits_2, cons_logits_2 = tower(**tower_args, dropout_probability=teacher_dropout_probability, name=name_scope)
    with ema_variable_scope("ema", var_scope, decay=ema_decay):
        class_logits_ema, cons_logits_ema = tower(**tower_args, dropout_probability=teacher_dropout_probability, name='ema')
        class_logits_ema, cons_logits_ema = tf.stop_gradient(class_logits_ema), tf.stop_gradient(cons_logits_ema)
    return (class_logits_1, cons_logits_1), (class_logits_2, cons_logits_2), (class_logits_ema, cons_logits_ema)


def tower(inputs,
          is_training,
          dropout_probability,
          input_noise,
          normalize_input,
          num_logits,
          is_initialization=False,
          name=None):
    num_classes = 2
    with tf.name_scope(name, "tower"):
        training_args = dict(
            is_training=is_training
        )
        # net = gussian_noise(inputs, scale = input_noise, is_training= is_training, name = 'gussian_noise')
        net = resnet_1(inputs = inputs,is_training= is_training)
        net = slim.flatten(net)
        primary_logits = fully_connected(net, num_classes, init=is_initialization)
        secondary_logits = fully_connected(net, num_classes, init=is_initialization)

        with tf.control_dependencies([tf.assert_greater_equal(num_logits, 1),
                                      tf.assert_less_equal(num_logits, 2)]):
            secondary_logits = tf.case([
                (tf.equal(num_logits, 1), lambda: primary_logits),
                (tf.equal(num_logits, 2), lambda: secondary_logits),
            ], exclusive=True, default=lambda: primary_logits)

        assert_shape(primary_logits, [None, num_classes])
        assert_shape(secondary_logits, [None, num_classes])
        return primary_logits, secondary_logits

# ----------------------------------------------------------------------------------------------------------------------

def errors(logits, labels, name=None):
    """Compute error mean and whether each unlabeled example is erroneous

    Assume unlabeled examples have label == -1.
    Compute the mean error over unlabeled examples.
    Mean error is NaN if there are no unlabeled examples.
    Note that unlabeled examples are treated differently in cost calculation.
    """
    with tf.name_scope(name, "errors") as scope:
        applicable = tf.not_equal(labels, -1)
        labels = tf.boolean_mask(labels, applicable)
        logits = tf.boolean_mask(logits, applicable)
        predictions = tf.argmax(logits, -1)
        labels = tf.cast(labels, tf.int64)
        per_sample = tf.to_float(tf.not_equal(predictions, labels))
        mean = tf.reduce_mean(per_sample, name=scope)
        return mean, per_sample


def classification_costs(logits, labels, name=None):
    """Compute classification cost mean and classification cost per sample

    Assume unlabeled examples have label == -1. For unlabeled examples, cost == 0.
    Compute the mean over all examples.
    Note that unlabeled examples are treated differently in error calculation.
    """
    with tf.name_scope(name, "classification_costs") as scope:
        applicable = tf.not_equal(labels, -1)

        # Change -1s to zeros to make cross-entropy computable
        labels = tf.where(applicable, labels, tf.zeros_like(labels))

        # This will now have incorrect values for unlabeled examples
        per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)

        # Retain costs only for labeled
        per_sample = tf.where(applicable, per_sample, tf.zeros_like(per_sample))

        # Take mean over all examples, not just labeled examples.
        labeled_sum = tf.reduce_sum(per_sample)
        total_count = tf.to_float(tf.shape(per_sample)[0])
        mean = tf.div(labeled_sum, total_count, name=scope)

        return mean, per_sample


def consistency_costs(logits1, logits2, cons_coefficient, mask, consistency_trust, name=None):


    with tf.name_scope(name, "consistency_costs") as scope:
        num_classes = 2
        assert_shape(logits1, [None, num_classes])
        assert_shape(logits2, [None, num_classes])
        assert_shape(cons_coefficient, [])
        softmax1 = tf.nn.softmax(logits1)
        softmax2 = tf.nn.softmax(logits2)

        kl_cost_multiplier = 2 * (1 - 1 / num_classes) / num_classes ** 2 / consistency_trust ** 2

        def pure_mse():
            costs = tf.reduce_mean((softmax1 - softmax2) ** 2, -1)
            return costs

        def pure_kl():
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits1, labels=softmax2)
            entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits2, labels=softmax2)
            costs = cross_entropy - entropy
            costs = costs * kl_cost_multiplier
            return costs

        def mixture_kl():
            with tf.control_dependencies([tf.assert_greater(consistency_trust, 0.0),
                                          tf.assert_less(consistency_trust, 1.0)]):
                uniform = tf.constant(1 / num_classes, shape=[num_classes])
                mixed_softmax1 = consistency_trust * softmax1 + (1 - consistency_trust) * uniform
                mixed_softmax2 = consistency_trust * softmax2 + (1 - consistency_trust) * uniform
                costs = tf.reduce_sum(mixed_softmax2 * tf.log(mixed_softmax2 / mixed_softmax1), axis=1)
                costs = costs * kl_cost_multiplier
                return costs

        costs = tf.case([
            (tf.equal(consistency_trust, 0.0), pure_mse),
            (tf.equal(consistency_trust, 1.0), pure_kl)
        ], default=mixture_kl)

        costs = costs * tf.to_float(mask) * cons_coefficient
        mean_cost = tf.reduce_mean(costs, name=scope)
        assert_shape(costs, [None])
        assert_shape(mean_cost, [])
        return mean_cost, costs


def total_costs(*all_costs, name=None):
    with tf.name_scope(name, "total_costs") as scope:
        for cost in all_costs:
            assert_shape(cost, [None])
        costs = tf.reduce_sum(all_costs, axis=1)
        mean_cost = tf.reduce_mean(costs, name=scope)
        return mean_cost, costs
