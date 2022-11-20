import tensorflow as tf
import tensorflow_models as tfm


class TensorflowModelsTest(tf.test.TestCase):

  def testVisionImport(self):
    _ = tfm.vision.layers.SqueezeExcitation(
        in_filters=8, out_filters=4, se_ratio=1)
    _ = tfm.vision.configs.image_classification.Losses()

  def testNLPImport(self):
    _ = tfm.nlp.layers.TransformerEncoderBlock(
        num_attention_heads=2, inner_dim=10, inner_activation='relu')
    _ = tfm.nlp.tasks.TaggingTask(params=tfm.nlp.tasks.TaggingConfig())

  def testCommonImports(self):
    _ = tfm.hyperparams.Config()
    _ = tfm.optimization.LinearWarmup(
        after_warmup_lr_sched=0.0, warmup_steps=10, warmup_learning_rate=0.1)


if __name__ == '__main__':
  tf.test.main()
