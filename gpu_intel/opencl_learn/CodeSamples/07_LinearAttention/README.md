# Linear attention

Refer: https://towardsdatascience.com/linear-attention-is-all-you-need-5fa9c845c1b5/

Attention forma:
```
Softmax(Q*K) * V
```

Linear Attention:
```
(Relu(K)*Relu(V)) * Q
```

默认的attention中的softmax本质上是归一化，或者相似度计算，理论上可以替换成Relu，因为Q*K,都是大矩阵运算，可以修改为先计算V*Ｑ，Linear Attention的计算复杂度变成了O(n).
