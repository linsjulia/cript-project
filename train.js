const tf = require('@tensorflow/tfjs');
// const tf = require('@tensorflow/tfjs-node'); // descomente se quiser acelerar
const { buildModel } = require('./model');
const dataset = require('./dataset.json');
const cliProgress = require('cli-progress');

// --- Codificação de caracteres ---
const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+/=";
const charToIndex = {};
for (let i = 0; i < charset.length; i++) charToIndex[charset[i]] = i + 1;

function encodeChars(s, maxLen = 64) {
  const arr = new Array(maxLen).fill(0);
  for (let i = 0; i < Math.min(s.length, maxLen); i++) {
    arr[i] = charToIndex[s[i]] || 0;
  }
  return arr;
}
module.exports.encodeChars = encodeChars;

// --- Labels ---
const labelMap = { plain: 0, hex: 1, base64: 2, md5: 3, sha1: 4, sha256: 5, rot13: 6 };

const seqLen = 64;
const vocabSize = charset.length + 1;
const X = [];
const y = [];

dataset.forEach(ex => {
  if (!(ex.label in labelMap)) throw new Error(`Label desconhecida: ${ex.label}`);
  X.push(encodeChars(String(ex.text || ''), seqLen));
  y.push(labelMap[ex.label]);
});

const xs = tf.tensor2d(X, [X.length, seqLen], 'int32');
let ys = tf.tensor1d(y.flat(), 'int32');

console.log('xs shape:', xs.shape, 'ys shape:', ys.shape);

async function runTraining() {
  const nClasses = Object.keys(labelMap).length;
  const epochs = 20;
  const batchSize = 64;
  const validationSplit = 0.2;

  const model = buildModel(vocabSize, seqLen, nClasses);

  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  const yOneHot = tf.tidy(() => tf.oneHot(ys.toInt(), nClasses).toFloat());

  // --- Configura barra de progresso ---
  const trainSamples = Math.floor(X.length * (1 - validationSplit));
  const stepsPerEpoch = Math.ceil(trainSamples / batchSize);
  const totalBatches = epochs * stepsPerEpoch;

  const bar = new cliProgress.SingleBar({
    format: 'Treinamento |{bar}| {percentage}% | Batch {value}/{total}',
    barCompleteChar: '\u2588',
    barIncompleteChar: '-',
    hideCursor: true
  });

  let processedBatches = 0;
  bar.start(totalBatches, 0);

  try {
    await model.fit(xs, yOneHot, {
      epochs,
      batchSize,
      validationSplit,
      callbacks: {
        onBatchEnd: async () => {
          processedBatches++;
          bar.update(processedBatches);
        },
        onEpochEnd: (epoch, logs) => {
          const acc = logs.acc ?? logs.accuracy ?? 0;
          const valAcc = logs.val_acc ?? logs.val_accuracy ?? 0;
          const loss = logs.loss ?? 0;
          const valLoss = logs.val_loss ?? 0;
          console.log(`\nEpoch ${epoch + 1}/${epochs} — loss=${loss.toFixed(4)} acc=${acc.toFixed(4)} val_loss=${valLoss.toFixed(4)} val_acc=${valAcc.toFixed(4)}`);
        },
        onTrainEnd: () => {
          bar.stop();
          console.log('Treinamento finalizado!');
        }
      }
    });

    await model.save('file://./model-saved');
    console.log('Modelo salvo em ./model-saved');
  } finally {
    xs.dispose();
    ys.dispose();
    yOneHot.dispose();
  }
}

if (require.main === module) {
  runTraining().catch(e => console.error('runTraining falhou:', e));
} else {
  module.exports.runTraining = runTraining;
}
