// model.js
const tf = require('@tensorflow/tfjs');


/**
 * Cria e compila o modelo.
 * Saída: softmax com nClasses unidades.
 * Loss: sparseCategoricalCrossentropy (labels como inteiros 1-D).
 */
function buildModel(vocabSize, seqLen, nClasses) {
  const input = tf.input({ shape: [seqLen], dtype: 'int32' }); // índices para embedding
  const emb = tf.layers.embedding({
    inputDim: vocabSize,
    outputDim: 32,
    inputLength: seqLen
  }).apply(input);

  const conv = tf.layers.conv1d({ filters: 64, kernelSize: 5, activation: 'relu' }).apply(emb);
  const pool = tf.layers.globalMaxPool1d().apply(conv);
  const dense1 = tf.layers.dense({ units: 64, activation: 'relu' }).apply(pool);
  const out = tf.layers.dense({ units: nClasses, activation: 'softmax' }).apply(dense1);

  const model = tf.model({ inputs: input, outputs: out });

  // Compilar com sparseCategoricalCrossentropy — espera labels inteiros 1-D
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'sparseCategoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

module.exports = { buildModel };
