const IMAGE = [[0,0,1,0,0],[0,1,1,1,0],[0,0,1,0,0],[0,1,1,1,0],[0,0,1,0,0]];
const ONES = [[1,1,1],[1,1,1],[1,1,1]];
const fmt = n => Number(n).toFixed(4);
const controls = `<div class="btnrow"><button class="cbtn" data-prev>← Back</button><button class="cbtn" data-next>Next →</button><button class="cbtn" data-reset>Reset</button></div>`;
function matrix(values, active = () => false) {
  return `<div class="ml-matrix" style="--cols:${values[0].length}">${values.flatMap((row,r) => row.map((v,c) => `<span class="ml-cell ${active(r,c) ? 'selected' : ''}">${v}</span>`)).join('')}</div>`;
}
function field(id, text, value, min, max, step=1) {
  return `<label>${text}<input data-field="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"><output data-value="${id}">${value}</output></label>`;
}
function connectFields(el, render) {
  el.querySelectorAll('input').forEach(input => input.addEventListener('input', () => {
    el.querySelector(`[data-value="${input.dataset.field}"]`).textContent = input.value;
    render();
  }));
  render();
}
function value(el, name) { return Number(el.querySelector(`[data-field="${name}"]`).value); }
function neuronDemo(el) {
  el.innerHTML = `<div class="ml-controls">${field('w','Weight w',0.5,-3,3,0.1)}${field('rate','Learning rate η',1,0,5,0.1)}</div><div class="ml-result" aria-live="polite"></div><p class="hint">Teaching example: x = 2, b = 0, target = 1. Only the weight is updated.</p>`;
  connectFields(el, () => {
    const r = ML.neuron(2,value(el,'w'),0,1,value(el,'rate'));
    el.querySelector('.ml-result').innerHTML = `<div class="flow"><span>z = ${fmt(r.z)}</span><b>→</b><span>σ(z) = ${fmt(r.y)}</span><b>→</b><span>L = ${fmt(r.loss)}</span></div><p><b>Backward:</b> dL/dw = (${fmt(r.y)} − 1) × ${fmt(r.y*(1-r.y))} × 2 = <strong>${fmt(r.dw)}</strong></p><p><b>Update:</b> w′ = ${fmt(r.nextW)} → y′ = ${fmt(r.nextY)} → L′ = <strong>${fmt(r.nextLoss)}</strong></p>`;
  });
}
function xorDemo(el) {
  let index = 0;
  el.innerHTML = `<div class="ml-result" aria-live="polite"></div>${controls}`;
  function render() {
    const [a,b] = [[0,0],[0,1],[1,0],[1,1]][index], h1=+(a+b>=0.5), h2=+(a+b>=1.5), y=+(h1-h2>=0.5);
    el.querySelector('.ml-result').innerHTML = `<p class="step-label">Input ${index+1} / 4</p><div class="flow"><span>x = [${a}, ${b}]</span><b>→</b><span>h = [${h1}, ${h2}]</span><b>→</b><span>XOR = ${y}</span></div><p>h₁ = step(x₁+x₂−0.5) = OR<br>h₂ = step(x₁+x₂−1.5) = AND<br>y = step(h₁−h₂−0.5)</p>`;
    el.querySelector('[data-prev]').disabled=index===0;
    el.querySelector('[data-next]').disabled=index===3;
  }
  el.querySelector('[data-prev]').onclick=()=>{index--;render();};
  el.querySelector('[data-next]').onclick=()=>{index++;render();};
  el.querySelector('[data-reset]').onclick=()=>{index=0;render();};render();
}
function convDemo(el) {
  let index=0;
  const output=ML.convolution(IMAGE,ONES);
  el.innerHTML=`<div class="ml-result" aria-live="polite"></div>${controls}<p class="hint">All filter weights = 1, bias = 0; stride 1, padding 0. This is cross-correlation, as used in CNN libraries.</p>`;
  function render() {
    const row=Math.floor(index/3),col=index%3;
    el.querySelector('.ml-result').innerHTML=`<p>Window ${index+1} / 9 · row ${row}, column ${col}</p><div class="matrix-flow"><div><small>Input 5 × 5</small>${matrix(IMAGE,(r,c)=>r>=row&&r<row+3&&c>=col&&c<col+3)}</div><div><small>Shared filter</small>${matrix(ONES)}</div><div><small>Output 3 × 3</small>${matrix(output.map((r,i)=>r.map((v,j)=>i*3+j<=index?v:'·')),(r,c)=>r===row&&c===col)}</div></div><p>Sum of this patch = <strong>${output[row][col]}</strong></p>`;
    el.querySelector('[data-prev]').disabled=index===0;el.querySelector('[data-next]').disabled=index===8;
  }
  el.querySelector('[data-prev]').onclick=()=>{index--;render();};el.querySelector('[data-next]').onclick=()=>{index++;render();};el.querySelector('[data-reset]').onclick=()=>{index=0;render();};render();
}
function shapeDemo(el) {
  el.innerHTML=`<div class="ml-controls">${field('n','Input size N',32,7,64)}${field('f','Filter size F',5,1,7)}${field('p','Padding P',2,0,4)}${field('s','Stride S',1,1,4)}${field('k','Filters K',10,1,32)}</div><div class="ml-result" aria-live="polite"></div>`;
  connectFields(el,()=>{
    const n=value(el,'n'),f=value(el,'f'),p=value(el,'p'),s=value(el,'s'),k=value(el,'k'),r=ML.shape(n,f,p,s,3,k);
    el.querySelector('.ml-result').innerHTML=`<p>Input channels C = 3</p><div class="flow"><span>Output<br><strong>${r.size} × ${r.size} × ${k}</strong></span><span>Parameters<br><strong>${r.parameters.toLocaleString()}</strong></span></div><p>floor((${n} + 2×${p} − ${f}) / ${s}) + 1 = ${r.size}</p><p>(${f}² × 3 + 1) × ${k} = ${r.parameters}</p>${(n+2*p-f)%s ? '<p class="callout note">The stride does not divide evenly. We use floor; the final incomplete window is skipped.</p>':''}`;
  });
}
function poolingDemo(el) {
  const input=[[1,3,2,4],[2,9,5,1],[0,1,8,2],[3,2,1,7]];
  el.innerHTML=`<div class="matrix-flow"><div><small>Input 4 × 4</small>${matrix(input,(r,c)=>r<2&&c<2)}</div><div><small>Max pool 2 × 2, stride 2</small>${matrix([[9,5],[3,8]],(r,c)=>r===0&&c===0)}</div></div><p>Highlighted patch: max(1, 3, 2, 9) = <strong>9</strong>. No trainable parameters.</p>`;
}
function rnnDemo(el) {
  let step=0,reverse=false;
  el.innerHTML=`<div class="ml-result" aria-live="polite"></div>${controls}<button class="cbtn" data-reverse>Reverse the inputs</button>`;
  function render() {
    const inputs=reverse?[-1,0,1]:[1,0,-1],states=ML.recurrent(inputs);
    el.querySelector('.ml-result').innerHTML=`<p>hₜ = tanh(0.5hₜ₋₁ + xₜ), h₀ = 0</p><div class="flow">${inputs.map((x,i)=>`<span class="${i<step?'done':''}">x${i+1} = ${x}<br>h${i+1} = ${i<step?fmt(states[i]):'?'}</span>`).join('<b>→</b>')}</div><p>${step===0?'Start with h₀ = 0. Press Next to consume one input.':`Step ${step}: tanh(0.5 × ${fmt(step===1?0:states[step-2])} + ${inputs[step-1]}) = ${fmt(states[step-1])}`}</p>`;
    el.querySelector('[data-prev]').disabled=step===0;el.querySelector('[data-next]').disabled=step===3;
  }
  el.querySelector('[data-prev]').onclick=()=>{step--;render();};el.querySelector('[data-next]').onclick=()=>{step++;render();};el.querySelector('[data-reset]').onclick=()=>{step=0;render();};el.querySelector('[data-reverse]').onclick=()=>{reverse=!reverse;step=0;render();};render();
}
function attentionDemo(el) {
  el.innerHTML=`<div class="ml-controls">${field('q1','Query component q₁',1,-3,3,0.1)}${field('q2','Query component q₂',0,-3,3,0.1)}</div><label class="mask-control"><input type="checkbox" data-mask> Causal mask: block key 2 (query at position 1)</label><div class="ml-result" aria-live="polite"></div><p class="hint">Keys [1,0], [0,1]; scalar values 2, 10. A teaching example, not a trained language model. Calculations use full precision; displayed values are rounded.</p>`;
  function render() {
    const r=ML.attention([value(el,'q1'),value(el,'q2')],[[1,0],[0,1]],[2,10],el.querySelector('[data-mask]').checked?0:null);
    el.querySelector('.ml-result').innerHTML=`<p>Scaled scores = [${r.scores.map(x=>Number.isFinite(x)?fmt(x):'−∞').join(', ')}]</p>${r.weights.map((w,i)=>`<div class="weight-row"><span>Key ${i+1}</span><div class="weight-track"><div style="width:${100*w}%"></div></div><b>${(100*w).toFixed(1)}%</b></div>`).join('')}<p>Weights sum to ${fmt(r.weights[0]+r.weights[1])}</p><p>Output = ${fmt(r.weights[0])} × 2 + ${fmt(r.weights[1])} × 10 = <strong>${fmt(r.output)}</strong></p>`;
  }
  connectFields(el,render);el.querySelector('[data-mask]').onchange=render;
}
function quizDemo(el) {
  const quiz=QUIZZES[LECTURE.number];
  el.innerHTML=quiz.map((q,i)=>`<fieldset class="ml-quiz"><legend>${i+1}. ${q.q}</legend>${q.options.map((o,j)=>`<label><input type="radio" name="quiz-${i}" value="${j}"> ${o}</label>`).join('')}<button class="cbtn" data-check="${i}">Check answer</button><p class="feedback" aria-live="polite"></p></fieldset>`).join('');
  el.querySelectorAll('[data-check]').forEach(btn=>btn.onclick=()=>{
    const i=Number(btn.dataset.check),holder=btn.closest('fieldset'),answer=holder.querySelector('input:checked');
    holder.querySelector('.feedback').textContent=!answer?'Choose an answer first.':Number(answer.value)===quiz[i].answer?`Correct. ${quiz[i].why}`:`Try again. ${quiz[i].why}`;
  });
}
const QUIZZES={
  9:[{q:'What does backpropagation compute?',options:['Updated weights','Gradients of the loss','New training labels'],answer:1,why:'The optimizer uses gradients to update parameters.'},{q:'Can a stack of affine layers represent XOR without nonlinear activations?',options:['Yes, with enough layers','No'],answer:1,why:'Composing affine layers still gives an affine function.'},{q:'For our x=2 example, dL/dw equals…',options:['dL/db','2 × dL/db','0'],answer:1,why:'The weight gradient contains the extra factor x.'}],
  10:[{q:'32×32×3 input, ten 5×5 filters, stride 1, padding 2: output?',options:['28×28×3','32×32×10','32×32×3'],answer:1,why:'Padding preserves spatial size; each filter creates one channel.'},{q:'How many parameters including biases?',options:['750','760','10240'],answer:1,why:'(5×5×3 + 1) × 10 = 760.'},{q:'Changing stride changes the number of learned filter weights.',options:['True','False'],answer:1,why:'Stride changes the positions visited, not the filter bank.'}],
  11:[{q:'Softmax of two equal finite scores is…',options:['[0,0]','[0.5,0.5]','[1,1]'],answer:1,why:'The two exponentials are equal and normalized by their sum.'},{q:'What does causal masking prevent?',options:['Using earlier tokens','Using future tokens','All attention'],answer:1,why:'Future key scores become −∞ before softmax.'},{q:'Does gradient clipping generally fix vanishing gradients?',options:['Yes','No'],answer:1,why:'Clipping caps large gradients; it does not restore tiny gradients.'}]
};
const DEMOS={neuron:neuronDemo,xor:xorDemo,convolution:convDemo,shape:shapeDemo,pooling:poolingDemo,rnn:rnnDemo,attention:attentionDemo,quiz:quizDemo};
