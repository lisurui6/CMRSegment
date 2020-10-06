import tensorflow as tf
from CMRSegment.common.constants import MODEL_DIR
print("reset")
tf.compat.v1.reset_default_graph()
model_path = str(MODEL_DIR.joinpath("3D", "biobank_low2high.ckpt-300"))
print("sess")

with tf.compat.v1.Session() as sess:
    print("run")
    sess.run(tf.compat.v1.global_variables_initializer())
    print("meta")
    # Import the computation graph and restore the variable values
    saver = tf.compat.v1.train.import_meta_graph('{0}.meta'.format(model_path))
    print("restore")
    saver.restore(sess, '{0}'.format(model_path))
