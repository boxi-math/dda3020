/* Numerical kernels shared by the live examples and the checks in tools/check.cjs. */
(function (root) {
  const sigmoid = z => z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  function neuron(x, w, b, target, rate) {
    const z = w * x + b, y = sigmoid(z), loss = (y - target) ** 2 / 2;
    const dz = (y - target) * y * (1 - y), dw = dz * x;
    const nextW = w - rate * dw, nextY = sigmoid(nextW * x + b);
    return {z, y, loss, dw, db: dz, nextW, nextY, nextLoss: (nextY - target) ** 2 / 2};
  }
  function shape(n, f, p, s, c, k) {
    return {size: Math.floor((n + 2 * p - f) / s) + 1, parameters: (f * f * c + 1) * k};
  }
  function convolution(image, kernel) {
    const h = image.length - kernel.length + 1, w = image[0].length - kernel[0].length + 1;
    return Array.from({length: h}, (_, r) => Array.from({length: w}, (_, c) =>
      kernel.reduce((sum, row, i) => sum + row.reduce((subtotal, value, j) =>
        subtotal + value * image[r + i][c + j], 0), 0)));
  }
  function recurrent(inputs, weight = 0.5) {
    let h = 0;
    return inputs.map(x => h = Math.tanh(weight * h + x));
  }
  function attention(query, keys, values, causalIndex = null) {
    const scores = keys.map((key, i) => causalIndex !== null && i > causalIndex ? -Infinity :
      key.reduce((sum, x, j) => sum + x * query[j], 0) / Math.sqrt(query.length));
    const max = Math.max(...scores), exp = scores.map(x => Math.exp(x - max));
    const total = exp.reduce((a, b) => a + b, 0), weights = exp.map(x => x / total);
    return {scores, weights, output: weights.reduce((sum, w, i) => sum + w * values[i], 0)};
  }
  function lstm(previous, forget, input, candidate, output) {
    const cell = forget * previous + input * candidate;
    return {cell, hidden: output * Math.tanh(cell)};
  }
  function gru(previous, x, reset, retain, weight = 0.7) {
    const candidate = Math.tanh(weight * reset * previous + x);
    return {candidate, hidden: retain * previous + (1-retain) * candidate};
  }
  function receptive(layers) {
    let r=1, j=1;
    return layers.map(([f,s]) => {r+=(f-1)*j;j*=s;return {field:r,spacing:j};});
  }
  const api = {sigmoid, neuron, shape, convolution, recurrent, attention, lstm, gru, receptive};
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ML = api;
})(globalThis);
