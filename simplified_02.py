import os, math, random, urllib.request
random.seed(42)

if not os.path.exists('input.txt'): urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt', 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
uchars = sorted(set(''.join(docs)))
BOS, vocab_size = len(uchars), len(uchars) + 1

class Value:
    def __init__(self, d, c=(), lg=()):
        self.data, self.grad, self._children, self._local_grads = d, 0, c, lg
    def __add__(self, o): o = o if isinstance(o, Value) else Value(o); return Value(self.data + o.data, (self, o), (1, 1))
    def __radd__(self, o): return self + o
    def __mul__(self, o): o = o if isinstance(o, Value) else Value(o); return Value(self.data * o.data, (self, o), (o.data, self.data))
    def __rmul__(self, o): return self * o
    def __pow__(self, o): return Value(self.data**o, (self,), (o * self.data**(o-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __sub__(self, o): return self + (-o)
    def __truediv__(self, o): return self * o**-1
    def backward(self):
        topo, vis = [], set()
        def build(v):
            if v not in vis:
                vis.add(v); [build(c) for c in v._children]; topo.append(v)
        build(self); self.grad = 1
        for v in reversed(topo):
            for c, g in zip(v._children, v._local_grads): c.grad += g * v.grad

n_embd, n_head, n_layer, block_size, lr, steps = 16, 4, 1, 8, 1e-2, 3
head_dim = n_embd // n_head
mat = lambda nout, nin, std=0.02: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': mat(vocab_size, n_embd), 'wpe': mat(block_size, n_embd), 'lm_head': mat(vocab_size, n_embd)}
for i in range(n_layer):
    for k, n, s in [('attn_wq', n_embd, 0.02), ('attn_wk', n_embd, 0.02), ('attn_wv', n_embd, 0.02), ('attn_wo', n_embd, 0), ('mlp_fc1', 4*n_embd, 0.02), ('mlp_fc2', n_embd, 0)]:
        state_dict[f'layer{i}.{k}'] = mat(n, 4*n_embd if k=='mlp_fc2' else n_embd, std=s)
params = [p for m in state_dict.values() for r in m for p in r]

def linear(x, w): return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
def softmax(l):
    m_v = max(v.data for v in l)
    exps = [(v - m_v).exp() for v in l]
    s = sum(exps); return [e / s for e in exps]
def rmsnorm(x):
    scale = (sum(xi * xi for xi in x) / len(x) + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(t_id, p_id, keys, values):
    x = rmsnorm([t + p for t, p in zip(state_dict['wte'][t_id], state_dict['wpe'][p_id])])
    for li in range(n_layer):
        r = x; x = rmsnorm(x)
        q, k, v = [linear(x, state_dict[f'layer{li}.attn_w{c}']) for c in 'qkv']
        keys[li].append(k); values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h, k_h, v_h = q[hs:hs+head_dim], [ki[hs:hs+head_dim] for ki in keys[li]], [vi[hs:hs+head_dim] for vi in values[li]]
            w = softmax([sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))])
            x_attn.extend([sum(w[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)])
        x = [a + b for a, b in zip(linear(x_attn, state_dict[f'layer{li}.attn_wo']), r)]
        r = x; x = rmsnorm(x)
        x = [a + b for a, b in zip(linear([xi.relu()**2 for xi in linear(x, state_dict[f'layer{li}.mlp_fc1'])], state_dict[f'layer{li}.mlp_fc2']), r)]
    return linear(x, state_dict['lm_head'])

m, v = [0.0]*len(params), [0.0]*len(params)
for step in range(steps):
    doc = docs[step % len(docs)]; tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens)-1); ks, vs, ls = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)], []
    for p_id in range(n): ls.append(-softmax(gpt(tokens[p_id], p_id, ks, vs))[tokens[p_id+1]].log())
    loss = (1/n) * sum(ls); loss.backward()
    lr_t = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
    for i, p in enumerate(params):
        m[i] = 0.9 * m[i] + 0.1 * p.grad; v[i] = 0.95 * v[i] + 0.05 * p.grad**2
        p.data -= lr_t * (m[i]/(1-0.9**(step+1))) / ((v[i]/(1-0.95**(step+1)))**0.5 + 1e-8); p.grad = 0
    print(f"step {step+1:2d} / {steps:2d} | loss {loss.data:.4f}")

for i in range(20):
    ks, vs, t_id, out = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)], BOS, []
    for p_id in range(block_size):
        probs = softmax([l / 0.5 for l in gpt(t_id, p_id, ks, vs)])
        t_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if t_id == BOS: break
        out.append(uchars[t_id])
    print(f"sample {i+1:2d}: {''.join(out)}")