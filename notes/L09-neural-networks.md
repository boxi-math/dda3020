# DDA3020 L09: Neural Networks

Complete study materials for the supplied Jicong Fan 2025 lecture, with full narrated parts, mathematical explanations and executable labs. These materials are not the new semester's official course announcement. PDF page references identify sources; new worked calculations are teaching examples. The original slides remain unchanged.

**Learning objectives:** Weighted sums and activation functions; Hidden representations and nonlinear separability; Local derivatives, chain rule and regularization; Graph branches and reverse-mode differentiation; Multilayer forward and backward passes; Optimization and numerical verification; Depth, approximation and generalization; Computational and parameter cost; Bridge from dense networks to convolution.

## Neurons and learned representations

Part 1 of 3. Follow the source lecture, then verify the computation.

### Watch the full part

<div class="card video" data-video="../videos/L09-1.mp4"></div>

### From choosing features to learning features

Welcome to Lecture Nine of Machine Learning. Our question is how to learn useful features rather than choose every feature by hand. In logistic regression, we apply a sigmoid to a weighted sum of an input vector. In a support vector machine, we choose a separating boundary, often using a kernel. Both approaches depend on how the original examples are represented. A poor representation can make even a good classifier struggle.

```text
Logistic regression
y = sigmoid(w^T x + b)
```

Imagine classifying handwritten digits. Raw pixels tell us brightness at individual positions. Features such as strokes and loops would be more directly useful, but designing them manually is difficult. A neural network learns intermediate quantities from the pixels, then uses those quantities to predict the label. We still specify the objective and optimization procedure. What changes is that some of the feature transformation becomes trainable along with the final decision rule.

```text
Fixed features -> classifier
Learned features -> classifier
```

This lecture has three parts. First we build neurons and hidden representations. Second we derive backpropagation through computational graphs. Third we connect multilayer learning, expressive power and computational cost to the motivation for convolution. By the end, you should be able to calculate a forward pass, explain why nonlinear activations matter, and derive gradients rather than memorize an update formula. Keep the distinction between representing a function and successfully learning it in mind throughout.

```text
Representation + decision
Train the composition
```

**Source:** L09 PDF pp. 1, 6.

### A mathematical neuron

The lecture motivates an artificial neuron with biological neurons receiving excitatory and inhibitory signals. This is an analogy, not an accurate model of brain function. Mathematically, a unit multiplies each input by a weight, adds the products and a bias, and applies an activation. A positive weight can increase the preactivation; a negative weight can decrease it. The bias shifts the response independently of the current input vector.

```text
z = w^T x + b
y = g(z)
```

Calculate one unit with two inputs. The input values are one and two. The weights are one half and minus one, and the bias is one. Multiplication gives one half and minus two. Adding the bias gives minus one half. This is the preactivation, usually called z. Notice that a negative preactivation does not yet determine the final output. We must know which activation function the model uses.

```text
x = [1, 2]
w = [0.5, -1], b = 1
z = -0.5
```

With a zero threshold step, minus one half produces zero. ReLU also returns zero here, but for positive inputs it returns the input value rather than a binary one. Sigmoid returns approximately zero point three seven seven five. These are different functions with different purposes. A perceptron commonly uses a hard decision; differentiable networks use activations suited to gradient training. The input layer supplies values; it need not perform this same computation.

```text
step(-0.5) = 0
ReLU(-0.5) = 0
sigmoid(-0.5) = 0.3775
```

Slide eleven gives a hard-sign model and then approximates its derivative by one for an update. This is a surrogate rule, not the true derivative of the sign function. The classical perceptron instead updates on mistakes. With labels plus or minus one, add learning rate times target times input to the weights, and learning rate times target to the bias. A smooth sigmoid model supports ordinary loss differentiation. Keep these training rules distinct: replacing an inconvenient derivative by one is an additional modeling choice, not a mathematical consequence of the hard threshold.

```text
Hard-step training:
Mistake: w <- w + eta*t*x
Bias: b <- b + eta*t
```

<div class="card demo" data-demo="neuron"><div class="head">Investigate the computation</div><div class="demo-body"></div></div>

**Source:** L09 PDF pp. 7, 11.

### Activations and Boolean decisions

The slides compare linear, ReLU, softplus, sigmoid and hyperbolic tangent activations. Softplus smoothly approximates the ReLU shape. Sigmoid has outputs between zero and one, while hyperbolic tangent has outputs between minus one and one. Saturated sigmoid units have small derivatives, which will matter during learning. ReLU is not differentiable exactly at zero; implementations select a convention there. A linear activation does not introduce the nonlinearity we need for richer decision boundaries.

```text
ReLU(z) = max(0,z)
sigmoid(z) = 1/(1+exp(-z))
```

A perceptron can implement familiar logic. For binary inputs, threshold the sum minus one point five to implement AND. Only two ones cross the threshold. Threshold the sum minus one half to implement OR. For NOT, threshold one half minus the input. We define the step to return one when its argument is nonnegative. None of these examples lands exactly on zero, so the boundary convention does not affect their truth tables.

```text
AND: step(x1+x2-1.5)
OR:  step(x1+x2-0.5)
NOT: step(0.5-x)
```

Now try exclusive OR, or XOR. Exactly one input should be one. The positive points occupy opposite corners of the unit square, and the negative points occupy the other two corners. No straight line separates them. Therefore a single perceptron cannot represent this function. Read the slide's statement about simple Boolean gates with this restriction: perceptrons represent linearly separable Boolean functions, not all possible binary logic.

```text
XOR: labels 0,1,1,0
No separating straight line
```

**Source:** L09 PDF pp. 10, 14.

### Hidden units solve XOR

Introduce two hidden units. The first computes OR and the second computes AND. Use the first hidden output with weight one, the second with weight minus one, and an output bias of minus one half. This reproduces the structure of the XOR construction in the lecture. We explicitly use binary step outputs because sign functions sometimes use minus one and one instead. Mixing these conventions would change the calculation.

```text
h1 = step(x1+x2-0.5)
h2 = step(x1+x2-1.5)
```

Trace the four inputs. Zero, zero maps to hidden values zero, zero and output zero. Zero, one and one, zero both map to one, zero and output one. One, one maps to one, one and output zero. In hidden coordinates, the final classes are linearly separable. A layer computes a function, so several layers compute a composition. This explains representation power. It does not prove that gradient descent can train hard step activations, whose derivatives are unhelpful for that purpose.

```text
Input [0,1] -> h = [1,0]
y = step(h1-h2-0.5) = 1
```

For a twenty eight by twenty eight digit image, flattening gives seven hundred and eighty four input values. A first-layer weight vector can be reshaped back into an image to inspect which pixel patterns it responds to. The slide illustrates a stroke-like detector. Hidden units need not acquire human-readable concepts, but their activations form a learned representation. The output layer combines that representation to distinguish digit classes. Later, convolution will preserve spatial neighborhoods instead of fully connecting every pixel to every hidden unit.

```text
Digit: 28 x 28 -> 784
Hidden units learn detectors
Output combines features
```

The next slide also gives a ReLU construction. Let the sum of the two inputs be s. Define hidden values as ReLU of s and ReLU of s minus one. The output ReLU of the first hidden value minus twice the second gives the exact zero-one XOR truth table. The displayed slide adds an extra negative one-half output bias, giving positive scores of one half instead of one. Those scores separate the classes, but a decision threshold is needed if the requested outputs are binary labels.

```text
ReLU version: s=x1+x2
h1=ReLU(s), h2=ReLU(s-1)
y=ReLU(h1-2*h2) gives 0,1,1,0
```

<div class="card demo" data-demo="xor"><div class="head">Investigate the computation</div><div class="demo-body"></div></div>

**Source:** L09 PDF pp. 15, 22.

### Predict, explain, then continue

**Predict before revealing the answer.**

Pause the video and calculate the hidden values and final prediction for input one, one. Then explain the result in words without using the term black box. As a second question, predict whether adding ten linear layers, with no nonlinear activation anywhere, can make a straight-line classifier represent XOR. Write down your answers before continuing. The aim is to reason about the computation, not recall a diagram.

<details><summary>Show the explanation</summary>

Both hidden units activate on one, one: OR is one and AND is one. Their contributions cancel, and the output threshold gives zero. Adding linear or affine layers without nonlinear activations still produces a single affine function, so it does not solve XOR. Multiplying matrices changes the coefficients but not the class of functions. We will return to this fact when discussing what depth contributes and what it does not contribute.

</details>

Remember three ideas. A mathematical neuron applies an activation to a weighted sum. Hidden layers can transform a difficult representation into one a final classifier can separate. Representing a function is different from finding its parameters through training. For practice, write all four rows of the XOR construction and check each threshold. In the next part, we keep the prediction smooth and work backward through its computation to learn the weights.

**Source:** L09 PDF pp. 12, 22.

## Backpropagation and computational graphs

Part 2 of 3. Follow the source lecture, then verify the computation.

### Watch the full part

<div class="card video" data-video="../videos/L09-2.mp4"></div>

### Cache the forward pass

Training starts by specifying a loss. Follow the lecture's scalar example: a weighted input plus bias goes through sigmoid, then we measure half the squared difference from the target. We use squared error here to match the slides. Classification networks often use cross entropy instead, and its gradient is different. Always derive from the objective actually being optimized. A formula memorized for another loss can give the wrong training rule.

```text
z = wx+b -> y = sigmoid(z)
L = (y-t)^2/2
```

Take input two, weight one half, bias zero and target one. The weighted sum is one, sigmoid gives approximately zero point seven three one zero five nine, and the loss is approximately zero point zero three six one six five. Store the preactivation and the prediction during the forward pass. A computational graph represents each input and intermediate quantity as a node, with edges showing direct dependencies. The graph makes reuse and branching explicit.

```text
x=2, w=0.5, b=0, t=1
z=1, y=0.731059
L=0.036165
```

We could expand the loss into a long expression and differentiate it symbolically. Backpropagation takes a more structured approach. Each node knows its local derivative, and we combine those local derivatives by the chain rule. Multiplication depends on both factors, addition passes sensitivity to both inputs, and sigmoid uses its cached output. This method applies to large compositions without writing an enormous closed-form derivative. The useful product is an efficient numerical derivative computation.

```text
Cache z and y
Differentiate local operations
Do not expand everything
```

**Source:** L09 PDF pp. 23, 29.

### Multiply local derivatives

Start at the loss and move backward. Differentiating half the squared error with respect to the prediction gives prediction minus target. The sigmoid derivative with respect to its input is prediction times one minus prediction. Therefore the loss sensitivity at the preactivation is the product of these two factors. This is an application of the chain rule, not a new learning assumption. At each step, write down which output is being differentiated with respect to which input.

```text
dL/dy = y-t
dy/dz = y(1-y)
```

The preactivation is weight times input plus bias. Its derivative with respect to the weight is the input value. Its derivative with respect to the bias is one. Consequently, the weight gradient has one extra input factor compared with the bias gradient. For our example the gradients are approximately minus zero point one zero five seven five four and minus zero point zero five two eight seven seven. Their negative signs indicate that small increases in these parameters decrease the loss locally.

```text
dL/dw = (y-t)y(1-y)x
dL/db = (y-t)y(1-y)
```

An optimizer now uses these gradients. If we update only the weight with learning rate one, it becomes approximately zero point six zero five seven five four. With the bias still zero, the next loss is approximately zero point zero two six three two zero. Updating both parameters would produce a different result, so state the convention explicitly. Backpropagation computes gradients; gradient descent changes parameters. A large learning rate can overshoot, and a negative gradient does not justify an arbitrarily large move.

```text
dL/dw = -0.105754
dL/db = -0.052877
w_new = w - eta*dL/dw
```

<div class="card demo" data-demo="neuron"><div class="head">Investigate the computation</div><div class="demo-body"></div></div>

**Source:** L09 PDF pp. 24, 30.

### Branches add; regularization adds a path

A graph is not always a single chain. If a variable affects the final result through several outgoing paths, its total gradient is the sum of those path contributions. The regularized loss on slide thirty one has a data-loss path and a penalty path. With penalty one half times the squared weight and coefficient lambda, add lambda times the weight to the data gradient. At weight one half and lambda zero point one, the extra contribution is zero point zero five.

```text
Lreg = L + lambda*w^2/2
dLreg/dw = dL/dw + lambda*w
```

The multivariate example in the lecture makes the same principle explicit. The final function is the second hidden quantity plus the exponential of the product of the two hidden quantities. The first hidden quantity is cosine of the input, and the second is the input squared. The original input influences the result through both hidden quantities. Differentiate the final function with respect to each hidden quantity separately, then multiply by each hidden quantity's derivative with respect to the input.

```text
f = h2 + exp(h1*h2)
h1=cos(x), h2=x^2
```

The first path contributes the exponential of their product, multiplied by the second hidden quantity and minus sine of the input. The second path contributes one plus the first hidden quantity times that exponential, multiplied by twice the input. Add the paths. In code, initialize intermediate gradient accumulators to zero and add incoming contributions. Replacing an earlier gradient contribution instead of adding the new contribution is a common bug. The same rule handles a shared weight used many times in a network.

```text
df/dx = exp(h1*h2)*h2*(-sin x)
       + (1+h1*exp(h1*h2))*2x
```

**Source:** L09 PDF pp. 30, 35.

### Reverse topological order

For a multilayer network, compute each layer's preactivation and activation in forward order. At a hidden layer, collect the sensitivity from all downstream units using the transpose of the next weight matrix, then multiply elementwise by the local activation derivative. The lecture writes this as sums over connected neurons. The matrix form is the same computation with dimensions made explicit. An output sensitivity must be derived from the chosen output activation and loss.

```text
a(l) = g(z(l))
z(l) = W(l)a(l-1)+b(l)
```

The weight gradient of a dense layer is an outer product: the current layer's preactivation sensitivity multiplied by the previous layer's activation transpose. The bias gradient is the current sensitivity. For a minibatch, sum or average examples consistently with how the loss is defined. The graph on slide thirty two combines a prediction loss with weight regularization. Read it operation by operation: matrix multiplication, activation, output projection, loss, and penalty. Branches must join by addition in the backward pass.

```text
delta(l) = (W(l+1)^T delta(l+1))
           * g'(z(l))
```

The general algorithm orders nodes so parents come before children. Compute their values in this topological order. Seed the final scalar output's sensitivity with one, then visit nodes in reverse order and propagate contributions to their parents. Compute gradients using the parameters from the same forward pass before updating them. Updating an early weight too soon can corrupt gradients that still depend on its old value. This reverse-mode process is what automatic differentiation systems automate.

```text
dL/dW(l) = delta(l) a(l-1)^T
dL/db(l) = delta(l)
Accumulate before updating
```

**Source:** L09 PDF pp. 32, 40.

### Predict the missing factor

**Predict before revealing the answer.**

Pause and predict. Keep the weight, bias, target and sigmoid objective, but set the input value to zero. What happens to the weight gradient? Must the bias gradient also become zero? Then consider a graph where the same variable feeds two multiplications. Should its backward contributions be added or averaged merely because there are two paths? Draw both paths before checking the answer. Graph structure, rather than a slogan, should determine your reasoning.

<details><summary>Show the explanation</summary>

The weight gradient is zero because changing the weight cannot change weight times a zero input. The bias still shifts the preactivation, so its gradient can remain nonzero. Contributions from multiple computational paths are added, not automatically averaged. Averaging is a separate choice in the definition of a batch loss. For practice, check our two analytic gradients using a central finite difference: perturb one parameter slightly up and down, evaluate the loss, and divide the loss difference by twice the perturbation.

</details>

Three rules summarize the part. Cache the intermediate values needed by local derivatives. Multiply derivatives along each path and add contributions where paths meet. Apply parameter updates after the gradient computation. Your exercise is to add the squared-weight penalty and verify its gradient numerically for several inputs. In the final part, we will apply these ideas to a small multilayer network and examine why deeper models can help, how much computation they cost, and why images motivate shared connections.

**Source:** L09 PDF pp. 24, 40.

## Multilayer learning, depth and the bridge to CNNs

Part 3 of 3. Follow the source lecture, then verify the computation.

### Watch the full part

<div class="card video" data-video="../videos/L09-3.mp4"></div>

### A complete two-layer numerical pass

Let us calculate an entire small network rather than only a scalar neuron. This is a new worked example illustrating the slide's multilayer rules. The input is one, two. The first weight matrix is the two by two identity, with zero biases. Both preactivations are positive, so ReLU passes through the values one and two. The hidden vector therefore equals the input here, but it still results from an explicit layer computation.

```text
x=[1,2]
W1=[[1,0],[0,1]], b1=[0,0]
h=ReLU(W1*x+b1)=[1,2]
```

The output weights are one and minus one, with zero output bias and a linear output. The prediction is one minus two, or minus one. The target is one, so the half squared-error loss is two. The loss derivative at the output is minus two. Multiplying this by the hidden vector gives output weight gradients minus two and minus four. The output bias gradient is minus two. Carefully distinguish this linear output from the sigmoid example in the preceding part.

```text
w2=[1,-1], b2=0
y=w2*h=-1
t=1, L=(y-t)^2/2=2
```

Propagate the output sensitivity through the output weights to get hidden sensitivities minus two and plus two. Both ReLU derivatives are one, because their preactivations are positive. Take the outer product with the input to obtain the first-layer gradient rows minus two, minus four and plus two, plus four. The first-layer bias gradients are minus two and plus two. Every displayed number can be checked with finite differences. The exercise code on the site performs those checks and a small training run.

```text
delta_out=-2
dL/dw2=[-2,-4]
dL/dW1=[[-2,-4],[2,4]]
```

<div class="card demo" data-demo="multilayer"><div class="head">Investigate the computation</div><div class="demo-body"></div></div>

**Source:** L09 PDF pp. 19, 22, 35, 39.

### What backpropagation costs

The lecture compares the cost of forward and backward passes. Multiplying an output-width by input-width matrix by a vector uses work proportional to the product of those widths. For a hidden layer with m outputs and d inputs, this is proportional to m times d. Bias additions and elementwise activations usually cost less than the dense multiplication. The output layer adds its own cost. Count the actual dimensions before reducing an expression to a single big O term.

```text
W: m x d, x: d x 1
Forward matrix-vector cost: O(md)
```

Backward propagation needs a transposed matrix multiplication to pass gradients toward the inputs and an outer product to form weight gradients. These have the same asymptotic order as the forward multiplication. The slide's statement that backward work is roughly twice forward work is a useful rough model, not a universal timing law. Implementation, batching, activation functions and hardware affect the constant. Training must also retain intermediate activations or pay extra computation to reconstruct them.

```text
Backward: input gradient + outer product
Same asymptotic order
Store or recompute activations
```

For many square dense hidden layers of width H, multiplying H by H gives a quadratic dependence on width, repeated across the depth. This explains why widening a dense network can become expensive. It does not imply every neural network's cost is the square of one width. Convolutions and other structured operators have different dimension formulas. When estimating memory, separate parameter storage, activation storage, gradients and optimizer state. Parameter count alone does not determine total training memory.

```text
Square width H, depth L:
Dense work roughly O(L H^2)
Not every layer is square
```

**Source:** L09 PDF pp. 40, 42.

### What depth adds and does not promise

A deep sequence of linear transformations collapses into their matrix product. Including biases still gives an affine transformation. Therefore stacking layers without nonlinearities does not by itself expand the represented function class. Add nonlinear activations and the conclusion changes. A composition of piecewise-linear ReLU functions can create many regions with different slopes. The network can build progressively useful intermediate features rather than just multiply all its weights into one fixed matrix.

```text
W3(W2(W1*x)) = W_eff*x
Affine layers still collapse
```

Universal approximation results say that, under suitable assumptions, a sufficiently wide nonlinear network can approximate a continuous target function on a compact domain arbitrarily well. This is an existence statement. It does not tell us how wide the network must be, how to find its parameters, or how many training examples it requires. The broad phrase any function on the slide needs these mathematical qualifications. Approximation ability is not equivalent to an optimization guarantee.

```text
Universal approximation:
Suitable nonlinearity, enough width
Continuous target on compact domain
```

Why then use depth? Some functions have efficient compositional representations, whereas a shallow network may require many more units for comparable approximation. The lecture also illustrates empirical generalization comparisons. These are motivation, not a theorem that adding depth always improves test performance. Data, regularization, architecture and optimization all matter. Diagnose training error separately from validation error: the first reflects fit to the observed examples, while the second provides evidence about predictions beyond those examples.

```text
Some functions need far less width
when represented compositionally
Generalization is not guaranteed
```

<div class="card demo" data-demo="xorTraining"><div class="head">Investigate the computation</div><div class="demo-body"></div></div>

**Source:** L09 PDF pp. 43, 47.

### Local connections and shared parameters

A large image reveals the limitation of fully connected layers. Even a modest hidden width must connect to every input pixel and channel. Flattening also discards the explicit arrangement of neighboring pixels, although that structure matters for edges and textures. The lecture introduces two ideas to reduce this burden: sparse local connections and shared parameters. These are architectural choices that encode useful assumptions about images, rather than universal rules for every learning problem.

```text
Dense image layer:
Every input -> every output
```

With local connectivity, one output depends only on a neighborhood of the previous layer. That neighborhood is its immediate receptive field. Multiple layers enlarge the region of the original input that can influence a later output. For two stride-one filters of width three, the interior receptive field has width five, not six, because adjacent windows overlap. Strides and pooling change the spacing between features, so we must track that spacing as well as the receptive-field size.

```text
Sparse connections:
Each output sees a local patch
Receptive field grows with depth
```

Parameter sharing reuses the same local weights at different positions. The lecture's one-dimensional sketch contrasts three shared filter weights with a larger fully connected connection pattern. In two-dimensional images, the corresponding kernel slides across width and height. Reusing a detector says that a useful pattern can occur at different positions. It does not mean different filters share all their weights with one another. In the next lecture we will count their activations, learned parameters and spatial dimensions carefully.

```text
Shared kernel weights
Same detector at many positions
Fewer learned parameters
```

**Source:** L09 PDF pp. 48, 52.

### Lecture Nine consolidation

**Predict before revealing the answer.**

Pause for the final prediction. Suppose we make our XOR network much deeper and wider, but replace every activation with the identity function. Could enough parameters eventually solve XOR exactly with one affine decision boundary? Next, explain why a network with enough representational capacity might still fail to learn from a particular training set. Your answer should identify at least two distinct causes rather than simply saying the model is not intelligent enough.

<details><summary>Show the explanation</summary>

The identity-activation network still collapses to an affine map, so extra layers and coefficients do not remove the XOR limitation. A capable nonlinear model can still fail because optimization does not find a useful solution, because data are insufficient or unrepresentative, or because generalization is poor. For practice, run the two-layer example, verify every gradient with finite differences, and then train on all four XOR points using smooth hidden activations. Compare training loss with classification accuracy.

</details>

The lecture has three connected lessons. Neural networks learn representations by composing weighted transformations and nonlinear activations. Backpropagation computes gradients through that composition using local rules and accumulated path contributions. Architectural structure controls expressive efficiency and computational cost. The site includes the complete derivations, numerical checks, exercises and answers. Next comes convolution: we will preserve spatial structure, reuse a filter, and calculate exactly how an image changes shape through a complete small network.

**Source:** L09 PDF pp. 1, 52.

## Summary, exercises and executable labs

Check every learning objective with an exercise, then verify the reference implementation.

### Self-check

<div class="card demo" data-demo="quiz"><div class="head">Check your understanding</div><div class="demo-body"></div></div>

### Exercises and worked solutions

**1. Trace — Weighted sums and activation functions.** Use x=[1,2], w=[0.5,-1], b=1. Calculate z, sigmoid(z), ReLU(z) and the zero-threshold step output.

<details><summary>Worked solution 1</summary><p>z=-0.5; sigmoid(z)=0.3775406688; ReLU(z)=0; step(z)=0.</p></details>

**2. Trace — Hidden representations and nonlinear separability.** Write all four truth-table rows for the OR/AND hidden-unit XOR network. Then evaluate the corrected ReLU construction.

<details><summary>Worked solution 2</summary><p>Hidden OR/AND rows: [0,0], [1,0], [1,0], [1,1]. Both constructions produce [0,1,1,0]. For ReLU use h1=ReLU(s), h2=ReLU(s-1), y=ReLU(h1-2h2). The slide p.18 subtracts another 0.5: its positive scores are 0.5 and need a label threshold.</p></details>

**3. Derive — Local derivatives, chain rule and regularization.** For x=2,w=0.5,b=0,t=1 and sigmoid squared-error loss, derive both gradients. Add lambda*w^2/2 with lambda=0.1.

<details><summary>Worked solution 3</summary><p>dL/dw=-0.1057541856; dL/db=-0.0528770928. The regularized weight gradient is -0.0557541856. Bias is not regularized in this objective.</p></details>

**4. Derive — Graph branches and reverse-mode differentiation.** Explain how to differentiate f=h2+exp(h1*h2), h1=cos(x), h2=x^2 without missing a branch.

<details><summary>Worked solution 4</summary><p>df/dx=exp(h1*h2)*h2*(-sin(x))+(1+h1*exp(h1*h2))*2*x. Add both paths from x. Their contributions are not averaged merely because the graph branches.</p></details>

**5. Trace — Multilayer forward and backward passes.** Compute every parameter gradient of the two-layer ReLU example in Part 3, including both biases.

<details><summary>Worked solution 5</summary><p>Prediction=-1, loss=2; output weight gradient=[-2,-4], output bias=-2; first weight gradient=[[-2,-4],[2,4]], first bias=[-2,2]. The lab checks all entries by central differences.</p></details>

**6. Implement — Optimization and numerical verification.** Run the finite-difference checks, then train the smooth XOR model. Record both the loss and the four probabilities.

<details><summary>Worked solution 6</summary><p>With NumPy, seed=7, hidden width=4, tanh hidden units, binary cross entropy and learning rate=0.5, the included run reaches loss about 0.002310 at step 2000 and predictions [0,1,1,0]. Re-run the supplied lab rather than assuming another initialization has the same trace.</p></details>

**7. Explain — Depth, approximation and generalization.** Why does a stack of affine layers collapse? State the scope of universal approximation and why it does not guarantee good test accuracy.

<details><summary>Worked solution 7</summary><p>Compose matrices and biases to obtain one affine map. Suitable nonlinear networks can approximate continuous functions on compact domains with sufficient width; this existence statement does not guarantee efficient representation, successful optimization, sufficient data or generalization.</p></details>

**8. Extend — Computational and parameter cost.** A dense layer has input width 100 and output width 200. Count parameters including biases. What changes if both widths double?

<details><summary>Worked solution 8</summary><p>100*200+200=20200. Doubling both widths gives 200*400+400=80400. Matrix work scales with the width product; bias costs scale only with output width.</p></details>

**9. Extend — Bridge from dense networks to convolution.** Derive the receptive-field width of two 3-wide stride-1 layers. Explain the two structural changes used by CNNs.

<details><summary>Worked solution 9</summary><p>The receptive field is 5. Local connectivity restricts each unit to a neighborhood; parameter sharing reuses a filter across spatial positions. They are separate architectural choices.</p></details>

### Run the labs

Download the [complete Python reference implementation](../labs/labs.py) and [the recorded numerical results](../labs/results.json). It requires Python 3 and NumPy. Running it checks every analytic derivative against central finite differences, trains the tiny XOR and character models with fixed seeds, and prints their final training losses. These are small educational models, not evidence of performance on unseen data.

```text
python3 labs.py
python3 labs.py --output ./lab-results
```

Gradient error must remain below 1e-7 for the supplied smooth test points. The ReLU gradient check deliberately uses positive preactivations, avoiding the nondifferentiable boundary. The convolution example verifies accumulation over shared spatial uses. The RNN implementation contains the full forward and BPTT loops rather than relying on an automatic differentiation library.

### Working with AI and further reading

Ask a tutor to identify tensor dimensions and local derivatives before suggesting code. Request a minimal counterexample for a claimed invariance or generalization guarantee. Verify every number with the reference lab and the interactive controls. Do not accept a smooth-looking training curve as proof of gradient correctness or test-set performance.

Perceptron 感知机; activation 激活函数; representation 表示; computational graph 计算图; backpropagation 反向传播; universal approximation 通用逼近; receptive field 感受野.

Further reading follows the original lecture references: Hornik, Stinchcombe and White (1989) on approximation; LeCun et al. (1998) on document recognition; Krizhevsky et al. (2012) on ImageNet; Hochreiter and Schmidhuber (1997) on LSTM; Cho et al. (2014) on GRU; Vaswani et al. (2017), Attention Is All You Need. These references provide context; the course calculations use the supplied decks and explicitly identified teaching examples.
