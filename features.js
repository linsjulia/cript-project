// features.js
const letters = [];
for (let i = 65; i <= 90; i++) letters.push(String.fromCharCode(i)); // A-Z
for (let i = 97; i <= 122; i++) letters.push(String.fromCharCode(i)); // a-z

function extractFeatures(s) {
  const counts = new Array(letters.length).fill(0);
  for (const ch of s) {
    const idx = letters.indexOf(ch);
    if (idx >= 0) counts[idx]++;
  }
  const len = Math.max(1, s.length);
  // normalizar por comprimento
  const normalized = counts.map(c => c / len);
  // adicionar comprimento normalizado (por exemplo /100)
  normalized.push(len / 100);
  return normalized; // length 52 + 1 = 53
}

module.exports = { extractFeatures, letters };
