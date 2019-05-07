import collections

type1 = collections.namedtuple('type1', ['attr1', 'attr2'])

a = type1(attr1=1, attr2=2)

print(a.attr1)

#==
Block = collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])

base_depth = 128
num_units = 5
stride = 2
b =Block('scope', 'bottomneck', [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': 1
  }] * (num_units - 1) + [{
      'depth': base_depth * 4,
      'depth_bottleneck': base_depth,
      'stride': stride
  }])


for i, unit in enumerate(b.args):
    print(i,unit)
