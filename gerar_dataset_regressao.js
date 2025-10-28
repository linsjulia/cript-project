// gerar_dataset_regressao.js
const fs = require('fs');

function randomString(len = 12) {
  const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ";
  let s = '';
  for (let i = 0; i < len; i++) s += chars[Math.floor(Math.random() * chars.length)];
  return s;
}

function rotN(s, n) {
  return s.replace(/[a-zA-Z]/g, c => {
    const base = c <= 'Z' ? 65 : 97;
    return String.fromCharCode((c.charCodeAt(0) - base + n) % 26 + base);
  });
}

const data = [];
const samples = 5000;
for (let i = 0; i < samples; i++) {
  const plain = randomString(6 + Math.floor(Math.random() * 10));
  const shift = Math.floor(Math.random() * 26);
  const enc = rotN(plain, shift);
  data.push({ text: enc, shift });
}
fs.writeFileSync('dataset_regression.json', JSON.stringify(data, null, 2));
console.log('dataset_regression.json gerado:', data.length);
