// train_regressor_skit.js
const fs = require('fs');
const sk = require('scikitjs');
const tf = require('@tensorflow/tfjs');

// define backend para ScikitJS
sk.setBackend(tf);

// importa função de features
const { extractFeatures } = require('./features');

// carregar dataset de regressão (ROT-N)
const raw = JSON.parse(fs.readFileSync('dataset_regression.json'));

// X: features, y: shift
const X = raw.map(r => extractFeatures(r.text));
const y = raw.map(r => r.shift);

// criar modelo LinearRegression
const { LinearRegression } = sk.linear_model;
const reg = new LinearRegression();

// treinar
reg.fit(X, y);

// salvar coeficientes e intercepto
const modelObj = {
  coef: reg.coef_.arraySync(),
  intercept: reg.intercept_.arraySync()
};

fs.writeFileSync('regressor_skit.json', JSON.stringify(modelObj, null, 2));
console.log('Regressor ScikitJS treinado e salvo em regressor_skit.json');
