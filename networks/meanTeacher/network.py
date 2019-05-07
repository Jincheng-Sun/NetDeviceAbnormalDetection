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
    preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
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
        net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
        # Convert end_points_collection into a dictionary of end_points.
        end_points = utils.convert_collection_to_dict(
            end_points_collection)

        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net,1 , name='pool5', keep_dims=True)
          end_points['global_pool'] = net
        if num_classes is not None:
          net = slim.conv1d(net, num_classes, 1, activation_fn=None,
                            normalizer_fn=None, scope='logits')
          end_points[sc.name + '/logits'] = net
          if spatial_squeeze:
            net = tf.squeeze(net, 1, name='SpatialSqueeze')
            end_points[sc.name + '/spatial_squeeze'] = net
          end_points['predictions'] = slim.softmax(net, scope='predictions')
        return net, end_points

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

def resnet_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 spatial_squeeze=True,
                 reuse=None,
                 scope='resnet_50'):
  """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
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


a = resnet_50(tf.placeholder(dtype = tf.float32, shape = [None,86,1], name = 'input'),num_classes=14)

with tf.Session() as sess:
    summary = tf.summary.FileWriter('log/',sess.graph)