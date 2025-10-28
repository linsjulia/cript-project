// server.js
const express = require('express');
const tf = require('@tensorflow/tfjs');
const bodyParser = require('body-parser');
const fs = require('fs');

const { encodeChars } = require('./train'); // função de codificação
const { extractFeatures } = require('./features');

const labelNames = ['plain', 'hex', 'base64', 'md5', 'sha1', 'sha256', 'rot13'];
const seqLen = 64;

let model;          // classificador
let regressor = null; // objeto ScikitJS carregado

const port = 3000;

async function start() {
  // 1️⃣ Carregar classificador
  model = await tf.loadLayersModel('file://./model-saved/model.json');
  console.log('Modelo classificador carregado.');

  // 2️⃣ Carregar regressor ScikitJS
  try {
    const data = JSON.parse(fs.readFileSync('regressor_skit.json'));
    regressor = data;
    console.log('Regressor ScikitJS carregado.');
  } catch (err) {
    console.warn('⚠️  Regressor ScikitJS não encontrado. Rode train_regressor_skit.js');
  }

  // 3️⃣ Criar app Express
  const app = express();
  app.use(bodyParser.json());
  app.use(express.static('public')); // servir index.html se quiser

  // 4️⃣ Rota /predict
  app.post('/predict', async (req, res) => {
    try {
      const text = String(req.body.text || '');
      const arr = encodeChars(text, seqLen);

      // Classificação
      const probs = tf.tidy(() => {
        const input = tf.tensor2d([arr], [1, seqLen], 'int32');
        const pred = model.predict(input);
        return pred.arraySync()[0];
      });

      const maxI = probs.indexOf(Math.max(...probs));
      const label = labelNames[maxI];
      const confidence = probs[maxI];

      // Regressão se ROT
      let predictedShift = null;
      if (label.startsWith('rot') && regressor) {
        const feats = extractFeatures(text);
        predictedShift = regressor.coef.reduce(
          (acc, c, i) => acc + c * feats[i],
          regressor.intercept
        );
        predictedShift = Math.round(Math.max(0, Math.min(25, predictedShift)));
      }

      res.json({
        label,
        confidence: confidence.toFixed(4),
        predictedShift,
        all: probs
      });
    } catch (err) {
      console.error('Erro em /predict:', err);
      res.status(500).json({ error: String(err) });
    }
  });

  app.listen(port, () => console.log(`Servidor rodando em http://localhost:${port}`));
}

start().catch(err => console.error('Erro ao iniciar servidor:', err));
