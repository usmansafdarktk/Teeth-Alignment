import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from models.solver import TSynNetSolver
from options.test_options import TestOptions


def test():
    opt = TestOptions().parse(save=False)
    opt.batch_size = 1   # test code only supports batchSize = 1
    with tf.Graph().as_default(), tf.Session() as sess:
        TSynNet = TSynNetSolver(sess, opt)
        TSynNet.test()


if __name__ == "__main__":
    test()