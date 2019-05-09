from networks.meanTeacher.model import Model
from networks.meanTeacher import minibatching
from networks.meanTeacher.dataset import EuTk


model = Model()
model['rampdown_length'] = 0
model['rampup_length'] = 5000
model['training_length'] = 40000
model['max_consistency_cost'] = 50.0

tensorboard_dir = model.save_tensorboard_graph()

eutk = EuTk()
print('finish loading dataset')
training_batches = minibatching.training_batches(eutk.training, n_labeled_per_batch=50)
evaluation_batches_fn = minibatching.evaluation_epoch_generator(eutk.evaluation)

model.train(training_batches, evaluation_batches_fn)
# eutk = EuTk()