/* Full-course experiments; training traces come from the executed Python labs. */
function multilayerDemo(el) {
  let step=0;
  const stages=[
    ['Forward: hidden layer','x=[1,2]; W₁=identity; b₁=[0,0].','z=[1,2] → ReLU(z)=[1,2].'],
    ['Forward: output and loss','w₂=[1,−1]; b₂=0; target=1.','y=−1; L=½(−1−1)²=2.'],
    ['Backward: output','δout = y−target = −2.','dL/dw₂=[−2,−4]; dL/db₂=−2.'],
    ['Backward: hidden','w₂ × δout = [−2,2]; both ReLU derivatives are 1.','dL/dW₁=[[-2,−4],[2,4]]; dL/db₁=[−2,2].'],
  ];
  el.innerHTML=`<div class="ml-result" aria-live="polite"></div>${controls}`;
  function render(){const s=stages[step];el.querySelector('.ml-result').innerHTML=`<p><b>${s[0]}</b></p><p>${s[1]}</p><div class="equation">${s[2]}</div><p>All parameter gradients are checked by central differences in the Python lab.</p>`;el.querySelector('[data-prev]').disabled=step===0;el.querySelector('[data-next]').disabled=step===3;}
  el.querySelector('[data-prev]').onclick=()=>{step--;render();};el.querySelector('[data-next]').onclick=()=>{step++;render();};el.querySelector('[data-reset]').onclick=()=>{step=0;render();};render();
}
function trainingDemo(el, kind) {
  const data=LAB_RESULTS[kind], trace=data.trace;
  el.innerHTML=`<div class="ml-controls">${field('snapshot','Recorded training snapshot',0,0,trace.length-1)}</div><div class="ml-result" aria-live="polite"></div><p class="hint">Actual saved run of the downloadable Python lab; this slider inspects recorded checkpoints rather than training a model in your browser.</p>`;
  connectFields(el,()=>{
    const r=trace[value(el,'snapshot')];
    const values=kind==='xor'?r.probabilities:r.probabilities.map((p,i)=>p[[1,2,2,3][i]]);
    const labels=kind==='xor'?['[0,0] → 1','[0,1] → 1','[1,0] → 1','[1,1] → 1']:['h → e','he → l','hel → l','hell → o'];
    el.querySelector('.ml-result').innerHTML=`<p><b>Step ${r.step}</b> · Training loss ${r.loss.toFixed(6)}</p>${values.map((p,i)=>`<div class="weight-row"><span>${labels[i]}</span><div class="weight-track"><div style="width:${100*p}%"></div></div><b>${(100*p).toFixed(1)}%</b></div>`).join('')}${kind==='language'?`<p>Greedy next characters at the known training prefixes: <code>${r.prediction}</code></p>`:'<p>Bars show P(class 1); targets are [0,1,1,0].</p>'}<p>Seed ${data.seed}; learning rate ${data.learning_rate}. Gradient-check maximum error ${data.gradient_error.toExponential(2)}.</p>`;
  });
}
function lstmDemo(el) {
  el.innerHTML=`<div class="ml-controls">${field('prev','Previous cell cₜ₋₁',1,-2,2,.1)}${field('f','Forget gate f',.8,0,1,.05)}${field('i','Input gate i',.25,0,1,.05)}${field('g','Candidate g',.4,-1,1,.05)}${field('o','Output gate o',.5,0,1,.05)}</div><div class="ml-result" aria-live="polite"></div><p class="hint">One cell component. Gates are supplied to isolate their effects; a trained LSTM computes them from learned projections.</p>`;
  connectFields(el,()=>{
    const r=ML.lstm(value(el,'prev'),value(el,'f'),value(el,'i'),value(el,'g'),value(el,'o'));
    el.querySelector('.ml-result').innerHTML=`<div class="equation">c = f × cprev + i × g = ${fmt(r.cell)}<br>h = o × tanh(c) = ${fmt(r.hidden)}</div><p>The cell carries memory; the hidden state is the gated visible output.</p>`;
  });
}
function gruDemo(el) {
  el.innerHTML=`<div class="ml-controls">${field('prev','Previous hidden state',.6,-1,1,.05)}${field('x','Input contribution',-.4,-1,1,.05)}${field('r','Reset gate r',.5,0,1,.05)}${field('z','Retention gate z',.75,0,1,.05)}</div><div class="ml-result" aria-live="polite"></div><p class="hint">Explicit convention: z is the fraction of the old hidden state retained. Reset multiplies the old state before the recurrent projection.</p>`;
  connectFields(el,()=>{
    const r=ML.gru(value(el,'prev'),value(el,'x'),value(el,'r'),value(el,'z'));
    el.querySelector('.ml-result').innerHTML=`<div class="equation">candidate = tanh(0.7 × r × hprev + x)<br>= ${fmt(r.candidate)}<br>h = z × hprev + (1−z) × candidate<br>= ${fmt(r.hidden)}</div>`;
  });
}
function slidePoolingDemo(el) {
  const input=LAB_RESULTS.convolution.slide_pool_input,out=LAB_RESULTS.convolution.slide_pool_output;
  let index=0;el.innerHTML=`<div class="ml-result" aria-live="polite"></div>${controls}`;
  function render(){const row=Math.floor(index/2),col=index%2;el.querySelector('.ml-result').innerHTML=`<div class="matrix-flow"><div><small>Actual slide p.62 input</small>${matrix(input,(r,c)=>Math.floor(r/2)===row&&Math.floor(c/2)===col)}</div><div><small>2×2, stride 2</small>${matrix(out,(r,c)=>r===row&&c===col)}</div></div><p>Selected patch maximum: <strong>${out[row][col]}</strong>. No trainable parameters.</p>`;el.querySelector('[data-prev]').disabled=index===0;el.querySelector('[data-next]').disabled=index===3;}
  el.querySelector('[data-prev]').onclick=()=>{index--;render();};el.querySelector('[data-next]').onclick=()=>{index++;render();};el.querySelector('[data-reset]').onclick=()=>{index=0;render();};render();
}
function receptiveDemo(el) {
  let index=0;const results=ML.receptive([[3,1],[2,2],[3,1]]),names=['CONV3 / stride1','POOL2 / stride2','CONV3 / stride1'];
  el.innerHTML=`<div class="ml-result" aria-live="polite"></div>${controls}`;
  function render(){el.querySelector('.ml-result').innerHTML=`<p>Start: r=1 and j=1. r′=r+(F−1)j; j′=jS.</p><div class="flow">${results.map((r,i)=>`<span class="${i<=index?'done':''}">${names[i]}<br>${i<=index?`r=${r.field}, j=${r.spacing}`:'?'}</span>`).join('<b>→</b>')}</div><p>Theoretical interior receptive field; padding changes border behavior.</p>`;el.querySelector('[data-prev]').disabled=index===0;el.querySelector('[data-next]').disabled=index===2;}
  el.querySelector('[data-prev]').onclick=()=>{index--;render();};el.querySelector('[data-next]').onclick=()=>{index++;render();};el.querySelector('[data-reset]').onclick=()=>{index=0;render();};render();
}
function architectureDemo(el) {
  el.innerHTML=`<div class="table-scroll"><table><thead><tr><th>Layer</th><th>Output shape</th><th>Parameters</th></tr></thead><tbody>${LAB_RESULTS.convolution.architecture.map(r=>`<tr><td>${r.layer}</td><td>${r.shape.join(' × ')}</td><td>${r.parameters}</td></tr>`).join('')}</tbody></table></div><p><b>Total: ${LAB_RESULTS.convolution.total_parameters} parameters.</b> ReLU and flatten add no learned parameters. Compare this table with your own calculation before using the reference code.</p>`;
}
Object.assign(DEMOS,{multilayer:multilayerDemo,xorTraining:el=>trainingDemo(el,'xor'),languageTraining:el=>trainingDemo(el,'language'),lstm:lstmDemo,gru:gruDemo,slidePooling:slidePoolingDemo,receptive:receptiveDemo,architecture:architectureDemo});
QUIZZES[9].push(
  {q:'A variable reaches the loss through two graph paths. Its gradient contributions are…',options:['Added','Automatically averaged','Only the larger is used'],answer:0,why:'The multivariable chain rule adds contributions from every path.'},
  {q:'Universal approximation guarantees gradient descent finds a good model.',options:['True','False'],answer:1,why:'It is an existence result under stated assumptions, not an optimization guarantee.'},
  {q:'A dense layer with 100 inputs and 200 outputs has how many parameters including biases?',options:['20000','20200','300'],answer:1,why:'100×200 weights plus 200 biases.'}
);
QUIZZES[10].push(
  {q:'2×2 stride-2 max pooling changes the channel count.',options:['True','False'],answer:1,why:'Each channel is processed independently.'},
  {q:'Two 3-wide, stride-1 convolutions have an interior receptive-field width of…',options:['3','5','6'],answer:1,why:'Each extra local layer expands the original dependency region by 2.'},
  {q:'Flattening a feature map adds learned parameters.',options:['True','False'],answer:1,why:'It rearranges activations. The following dense layer has the learned weights.'}
);
QUIZZES[11].push(
  {q:'Truncated BPTT can carry a hidden-state value while detaching its old gradient graph.',options:['True','False'],answer:0,why:'Preserving a value differs from preserving its backward history.'},
  {q:'For LSTM f=1 and i=0, the new cell state is…',options:['Zero','The previous cell','Always the hidden state'],answer:1,why:'The old cell is retained and no candidate content is written.'},
  {q:'Cross-attention queries and keys must have the same sequence length.',options:['True','False'],answer:1,why:'Target queries can attend to a different-length source, creating a rectangular map.'}
);
