import os, math, random, urllib.request
random.seed(42)

if not os.path.exists('input.txt'): 
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt', 'input.txt')
docs = [line.strip() for line in open('input.txt').read().strip().split('\n') if line.strip()]
random.shuffle(docs)
uchars = sorted(set(''.join(docs)))
BOS, vocab_size = len(uchars), len(uchars) + 1

class Value:
    def __init__(self, data, children=(), local_grads=()):
        self.data, self.grad, self._children, self._local_grads = data, 0, children, local_grads
    def __add__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))
    def __radd__(self, other): return self + other
    def __mul__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __rmul__(self, other): return self * other
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * other**-1
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v); [build(child) for child in v._children]; topo.append(v)
        build(self); self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads): child.grad += local_grad * v.grad

num_steps = 30 # Set to 500 for full training
n_embd, n_head, n_layer, block_size, lr = 16, 4, 1, 8, 1e-2
head_dim = n_embd // n_head
matrix = lambda nout, nin, std=0.02: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    for k, n, s in [('wq', n_embd, 0.02), ('wk', n_embd, 0.02), ('wv', n_embd, 0.02), ('wo', n_embd, 0), ('fc1', 4*n_embd, 0.02), ('fc2', n_embd, 0)]:
        state_dict[f'layer{i}.{"attn_" if "w" in k else "mlp_"}{k}'] = matrix(n, 4*n_embd if k=='fc2' else n_embd, std=s)
params = [p for mat in state_dict.values() for row in mat for p in row]

def linear(x, w): return [sum(wi * xi for wi, xi in zip(weight_row, x)) for weight_row in w]
def softmax(logits):
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    sum_exps = sum(exps); return [e / sum_exps for e in exps]
def rmsnorm(x):
    scale = (sum(xi * xi for xi in x) / len(x) + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    x = rmsnorm([t + p for t, p in zip(state_dict['wte'][token_id], state_dict['wpe'][pos_id])])
    for li in range(n_layer):
        res = x; x = rmsnorm(x)
        q, k, v = [linear(x, state_dict[f'layer{li}.attn_w{c}']) for c in 'qkv']
        keys[li].append(k); values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h, k_h, v_h = q[hs:hs+head_dim], [ki[hs:hs+head_dim] for ki in keys[li]], [vi[hs:hs+head_dim] for vi in values[li]]
            w = softmax([sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))])
            x_attn.extend([sum(w[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)])
        x = [a + b for a, b in zip(linear(x_attn, state_dict[f'layer{li}.attn_wo']), res)]
        res = x; x = rmsnorm(x)
        x = [a + b for a, b in zip(linear([xi.relu()**2 for xi in linear(x, state_dict[f'layer{li}.mlp_fc1'])], state_dict[f'layer{li}.mlp_fc2']), res)]
    return linear(x, state_dict['lm_head'])

m, v = [0.0]*len(params), [0.0]*len(params)
for step in range(num_steps):
    doc = docs[step % len(docs)]; tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens)-1); ks, vs, losses = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)], []
    for p_id in range(n): losses.append(-softmax(gpt(tokens[p_id], p_id, ks, vs))[tokens[p_id+1]].log())
    loss = (1/n) * sum(losses); loss.backward()
    lr_t = lr * 0.5 * (1 + math.cos(math.pi * step / num_steps))
    for i, p in enumerate(params):
        m[i] = 0.9 * m[i] + 0.1 * p.grad; v[i] = 0.95 * v[i] + 0.05 * p.grad**2
        p.data -= lr_t * (m[i]/(1-0.9**(step+1))) / ((v[i]/(1-0.95**(step+1)))**0.5 + 1e-8); p.grad = 0
    print(f"step {step+1:2d} / {num_steps:2d} | loss {loss.data:.4f}")

for i in range(20):
    ks, vs, token_id, out = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)], BOS, []
    for p_id in range(block_size):
        probs = softmax([logit / 0.5 for logit in gpt(token_id, p_id, ks, vs)])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS: break
        out.append(uchars[token_id])
    print(f"sample {i+1:2d}: {''.join(out)}")