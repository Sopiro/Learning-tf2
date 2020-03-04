import tensorflow as tf

test = tf.Variable(0)

ckpt = tf.train.Checkpoint(test=test)
manager = tf.train.CheckpointManager(ckpt, '../ckpt_test', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

print(test.numpy())
test.assign_add(1)

manager.save()
