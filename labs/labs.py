"""Runnable reference labs. Requires NumPy; run this file to verify and export results."""
import json
import math
from pathlib import Path

import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def gradient_check(parameters, loss_gradient, eps=1e-5):
    """Compare every parameter's analytic derivative with a central difference."""
    _, gradients = loss_gradient(parameters)
    error = 0.0
    for name, array in parameters.items():
        for index in np.ndindex(array.shape):
            original = array[index]
            try:
                array[index] = original + eps
                plus, _ = loss_gradient(parameters)
                array[index] = original - eps
                minus, _ = loss_gradient(parameters)
            finally:
                array[index] = original
            error = max(error, abs((plus-minus)/(2*eps)-gradients[name][index]))
    return float(error)


def two_layer(parameters):
    x = np.array([1., 2.])
    z = parameters['W1'] @ x + parameters['b1']
    h = np.maximum(z, 0)
    y = float(parameters['w2'] @ h + parameters['b2'][0])
    dy = y - 1
    dz = parameters['w2'] * dy * (z > 0)
    return dy*dy/2, dict(W1=np.outer(dz, x), b1=dz,
                        w2=dy*h, b2=np.array([dy]))


def xor_lab():
    x = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    target = np.array([[0.],[1.],[1.],[0.]])
    rng = np.random.default_rng(7)
    p = dict(W=rng.normal(0,.5,(2,4)), b=np.zeros(4),
             v=rng.normal(0,.5,(4,1)), c=np.zeros(1))

    def objective(parameters):
        h = np.tanh(x @ parameters['W'] + parameters['b'])
        logits = h @ parameters['v'] + parameters['c']
        prediction = 1/(1+np.exp(-logits))
        loss = np.mean(np.logaddexp(0,logits)-target*logits)
        dz = (prediction-target)/4
        dh = (dz @ parameters['v'].T)*(1-h*h)
        return float(loss), dict(W=x.T @ dh, b=dh.sum(axis=0),
                                 v=h.T @ dz, c=dz.sum(axis=0))
    error=gradient_check(p,objective)
    trace=[]
    for step in range(2001):
        loss, gradients=objective(p)
        if step in (0,10,50,100,300,1000,2000):
            h=np.tanh(x@p['W']+p['b'])
            probabilities=(1/(1+np.exp(-(h@p['v']+p['c'])))).ravel()
            trace.append(dict(step=step,loss=loss,probabilities=probabilities.tolist()))
        if step < 2000:
            for name in p:
                p[name]-=.5*gradients[name]
    assert trace[-1]['loss']<.03
    assert [int(v>=.5) for v in trace[-1]['probabilities']]==[0,1,1,0]
    return dict(gradient_error=error,trace=trace,seed=7,learning_rate=.5)


def convolution_lab():
    image=np.array([[0.,0.,1.,0.,0.],[0.,1.,1.,1.,0.],[0.,0.,1.,0.,0.],
                    [0.,1.,1.,1.,0.],[0.,0.,1.,0.,0.]])
    def apply(kernel):
        return np.array([[np.sum(image[r:r+3,c:c+3]*kernel)
                          for c in range(3)] for r in range(3)])
    upstream=np.arange(1.,10.).reshape(3,3)/9
    def objective(parameters):
        output=apply(parameters['kernel'])
        dk=np.zeros((3,3))
        for r in range(3):
            for c in range(3):
                dk+=upstream[r,c]*image[r:r+3,c:c+3]
        return float(np.sum(upstream*output)),dict(kernel=dk)
    parameters=dict(kernel=np.ones((3,3)))
    error=gradient_check(parameters,objective)
    output=apply(parameters['kernel'])
    assert output.tolist()==[[4.,5.,4.],[5.,7.,5.],[4.,5.,4.]]
    grid=np.array([[1,1,2,4],[5,6,7,8],[3,2,1,0],[1,2,3,4]])
    pool=[[int(grid[r:r+2,c:c+2].max()) for c in (0,2)] for r in (0,2)]
    assert pool==[[6,8],[3,4]]
    return dict(output=output.tolist(),kernel_gradient=objective(parameters)[1]['kernel'].tolist(),
                gradient_error=error,slide_pool_input=grid.tolist(),slide_pool_output=pool,
                architecture=[dict(layer='conv1',shape=[32,32,8],parameters=224),
                              dict(layer='pool1',shape=[16,16,8],parameters=0),
                              dict(layer='conv2',shape=[16,16,16],parameters=1168),
                              dict(layer='pool2',shape=[8,8,16],parameters=0),
                              dict(layer='dense',shape=[10],parameters=10250)],total_parameters=11642)


def language_lab():
    """A complete 4-hidden-unit character RNN, including analytic BPTT."""
    rng=np.random.default_rng(7)
    inputs=[0,1,2,2];targets=[1,2,2,3]
    p=dict(Wx=rng.normal(0,.2,(4,4)),Wh=rng.normal(0,.2,(4,4)),
           Wy=rng.normal(0,.2,(4,4)),bh=np.zeros(4),by=np.zeros(4))

    def forward(parameters):
        states=[np.zeros(4)];probabilities=[]
        for symbol in inputs:
            states.append(np.tanh(parameters['Wh']@states[-1]+parameters['Wx'][:,symbol]+parameters['bh']))
            probabilities.append(softmax(parameters['Wy']@states[-1]+parameters['by']))
        return states,np.array(probabilities)

    def objective(parameters):
        states,probabilities=forward(parameters)
        loss=-np.log(probabilities[np.arange(4),targets]).mean()
        gradients={name:np.zeros_like(value) for name,value in parameters.items()}
        future=np.zeros(4)
        for t in reversed(range(4)):
            logits_gradient=probabilities[t].copy()
            logits_gradient[targets[t]]-=1
            logits_gradient/=4
            gradients['Wy']+=np.outer(logits_gradient,states[t+1])
            gradients['by']+=logits_gradient
            state_gradient=parameters['Wy'].T@logits_gradient+future
            dz=state_gradient*(1-states[t+1]**2)
            gradients['Wx'][:,inputs[t]]+=dz
            gradients['Wh']+=np.outer(dz,states[t])
            gradients['bh']+=dz
            future=parameters['Wh'].T@dz
        return float(loss),gradients

    error=gradient_check(p,objective)
    trace=[]
    for step in range(601):
        loss,gradients=objective(p)
        if step in (0,10,50,100,300,600):
            _,probabilities=forward(p)
            trace.append(dict(step=step,loss=loss,probabilities=probabilities.tolist(),
                              prediction=''.join('helo'[i] for i in probabilities.argmax(axis=1))))
        if step<600:
            norm=math.sqrt(sum(float(np.sum(g*g)) for g in gradients.values()))
            factor=min(1.,1./max(norm,1e-12))
            for name in p:
                p[name]-=.5*factor*gradients[name]
    generated='h'
    state=np.zeros(4)
    symbol=0
    for _ in range(4):
        state=np.tanh(p['Wh']@state+p['Wx'][:,symbol]+p['bh'])
        symbol=int(softmax(p['Wy']@state+p['by']).argmax())
        generated+='helo'[symbol]
    assert generated=='hello'
    assert trace[-1]['prediction']=='ello' 
    assert trace[-1]['loss']<.03
    return dict(gradient_error=error,trace=trace,vocabulary=list('helo'),
                inputs=list('hell'),targets=list('ello'),hidden_width=4,seed=7,generated=generated,
                learning_rate=.5,global_norm_clip=1,parameters=56)


def verify(output_dir=None):
    parameters=dict(W1=np.eye(2),b1=np.zeros(2),w2=np.array([1.,-1.]),b2=np.zeros(1))
    loss,gradients=two_layer(parameters)
    assert loss==2
    assert gradients['W1'].tolist()==[[-2.,-4.],[2.,4.]]
    results=dict(two_layer=dict(loss=loss,gradient_error=gradient_check(parameters,two_layer),
                                gradients={name:array.tolist() for name,array in gradients.items()}),
                 xor=xor_lab(),convolution=convolution_lab(),language=language_lab(),
                 lstm=dict(cell=.8*1+.25*.4,hidden=.5*math.tanh(.9)),
                 gru=dict(previous=.6,retention=.75,candidate=-.2,result=.75*.6+.25*(-.2)))
    for name in ('two_layer','xor','convolution','language'):
        assert results[name]['gradient_error']<1e-7,(name,results[name]['gradient_error'])
    assert sum(row['parameters'] for row in results['convolution']['architecture'])==11642
    if output_dir is not None:
        target=Path(output_dir)
        target.mkdir(parents=True,exist_ok=True)
        (target/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        (target/'results.js').write_text('const LAB_RESULTS = '+json.dumps(results)+';\n')
    print(json.dumps({name:results[name]['gradient_error'] for name in ('two_layer','xor','convolution','language')}))
    print('XOR final loss:',results['xor']['trace'][-1]['loss'])
    print('RNN final loss:',results['language']['trace'][-1]['loss'])
    print('Autoregressive greedy generation:',results['language']['generated'])
    return results


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,help="Optional output directory for JSON and JS training traces")
    verify(parser.parse_args().output)
