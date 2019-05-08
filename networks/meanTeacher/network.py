import logging
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.layers.python.layers import utils
from networks.meanTeacher.resnet_utils import subsample, conv1d_same, stack_blocks_dense, Block
LOG = logging.getLogger('main')

@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
  with tf.variable_scope(scope, 'bottleneck', [inputs]) as sc:
    depth_in = utils.last_dimension(inputs.get_shape(), min_rank=3)
    # preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact',trainable=True,)
    preact = tf.keras.layers.BatchNormalization()(inputs)
    preact = tf.nn.relu(preact)
    # preact = tf.nn.batch_normalization(inputs,scope = 'batch_norm',training =True)
    # preact = tf.nn.relu(preact,name='act')

    if depth == depth_in:
      shortcut = subsample(inputs, stride, 'shortcut')
    else:
      shortcut = slim.conv1d(preact, depth, 1, stride=stride,
                             normalizer_fn=None, activation_fn=None,
                             scope='shortcut')

    residual = slim.conv1d(preact, depth_bottleneck, 1, stride=1,
                           scope='conv1')
    residual = slim.conv1d(residual, depth_bottleneck, 3, stride=stride, padding='SAME', scope='conv2')
    # residual = subsample(residual, factor=stride)
    residual = slim.conv1d(residual, depth, 1, stride=1,
                           normalizer_fn=None, activation_fn=None,
                           scope='conv3')

    output = shortcut + residual

    return utils.collect_named_outputs(outputs_collections,
                                            sc.name,
                                            output)


def resnet(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              reuse=None,
              scope=None):
  with tf.variable_scope(scope, 'resnet', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    with slim.arg_scope([slim.conv1d, bottleneck,
                         stack_blocks_dense],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        net = inputs
        if include_root_block:
          if output_stride is not None:
            if output_stride % 4 != 0:
              raise ValueError('The output_stride needs to be a multiple of 4.')
            output_stride /= 4
          # We do not include batch normalization or activation functions in
          # conv1 because the first ResNet unit will perform these. Cf.
          # Appendix of [2].
          with slim.arg_scope([slim.conv1d],
                              activation_fn=None, normalizer_fn=None):
            net = slim.conv1d(net, 32, 7, stride=1, scope='conv1')
        net = stack_blocks_dense(net, blocks, output_stride)
        # This is needed because the pre-activation variant does not have batch
        # normalization or activation functions in the residual unit output. See
        # Appendix of [2].
        # net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.nn.relu(net)
        # Convert end_points_collection into a dictionary of end_points.
        end_points = utils.convert_collection_to_dict(
            end_points_collection)

        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net,1 , name='pool5', keep_dims=True)
          end_points['global_pool'] = net
        # if num_classes is not None:
        #   net = slim.conv1d(net, num_classes, 1, activation_fn=None,
        #                     normalizer_fn=None, scope='logits')
        #   end_points[sc.name + '/logits'] = net
        #   if spatial_squeeze:
        #     net = tf.squeeze(net, 1, name='SpatialSqueeze')
        #     end_points[sc.name + '/spatial_squeeze'] = net
        #   end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net

def resnet_block(scope, base_depth, num_units, stride):

  return Block(scope, bottleneck, [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])
# resnet.default_image_size = 224

def resnet_1(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_1'):
  blocks = [
      resnet_block('block1', base_depth=32, num_units=2, stride=2),
      resnet_block('block2', base_depth=64, num_units=4, stride=2),
      resnet_block('block3', base_depth=128, num_units=4, stride=2),
      resnet_block('block4', base_depth=256, num_units=4, stride=1),
  ]
  return resnet(inputs, blocks, num_classes, is_training=is_training,
                   global_pool=global_pool, output_stride=output_stride,
                   include_root_block=True, spatial_squeeze=spatial_squeeze,
                   reuse=reuse, scope=scope)


# a = resnet_1(tf.placeholder(dtype = tf.float32, shape = [None,86,1], name = 'input'),num_classes=14)
#
# with tf.Session() as sess:
#     summary = tf.summary.FileWriter('log/',sess.graph)

@slim.add_arg_scope
def fully_connected(inputs, num_outputs,
                    activation_fn=None, init_scale=1., init=False,
                    eval_mean_ema_decay=0.999, is_training=None, scope=None):
    #pylint: disable=invalid-name
    with tf.variable_scope(scope, "fully_connected"):
        if is_training is None:
            is_training = tf.constant(True)
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V',
                                [int(inputs.get_shape()[1]), num_outputs],
                                tf.float32,
                                tf.random_normal_initializer(0, 0.05),
                                trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
            x_init = tf.matmul(inputs, V_norm)
            m_init, v_init = tf.nn.moments(x_init, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=tf.zeros_like(scale_init), trainable=True)
            x_init = tf.reshape(
                scale_init, [1, num_outputs]) * (x_init - tf.reshape(m_init, [1, num_outputs]))
            if activation_fn is not None:
                x_init = activation_fn(x_init)
            return x_init
        else:
            V, b = [tf.get_variable(var_name) for var_name in ['V', 'b']]

            # use weight normalization (Salimans & Kingma, 2016)
            inputs = tf.matmul(inputs, V)
            training_mean = tf.reduce_mean(inputs, [0])

            with tf.name_scope("eval_mean") as var_name:
                # Note that:
                # - We do not want to reuse eval_mean, so we take its name from the
                #   current name_scope and create it directly with tf.Variable
                #   instead of using tf.get_variable.
                # - We initialize with zero to avoid initialization order difficulties.
                #   Initializing with training_mean would probably be better.
                eval_mean = tf.Variable(tf.zeros(shape=training_mean.get_shape()),
                                        name=var_name,
                                        dtype=tf.float32,
                                        trainable=False)

            def _eval_mean_update():
                difference = (1 - eval_mean_ema_decay) * (eval_mean - training_mean)
                return tf.assign_sub(eval_mean, difference)

            def _no_eval_mean_update():
                "Do nothing. Must return same type as _eval_mean_update."
                return eval_mean

            eval_mean_update = tf.cond(is_training, _eval_mean_update, _no_eval_mean_update)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, eval_mean_update)
            mean = tf.cond(is_training, lambda: training_mean, lambda: eval_mean)
            inputs = inputs - mean
            inputs = inputs + tf.reshape(b, [1, num_outputs])

            # apply nonlinearity
            if activation_fn is not None:
                inputs = activation_fn(inputs)
            return inputs