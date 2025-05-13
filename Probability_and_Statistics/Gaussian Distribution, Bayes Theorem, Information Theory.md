정규분포는 다음과 같은 형태를 가짐
$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$
- 여기서 $\mu$는 평균, $\sigma^2$는 분산을 의미함

# 정규 분포가 등장하는 주요 사례
## Weight Initialization
이미지 생성에 있어서는 초기화가 매우 중요함. 본인도 딥러닝 수업 때 GAN 실습을 진행하며 일반 Gaussian Distribution으로 초기화했을 때는 잘 작동하지 않았고, Xavier Initialization으로 초기화했을 때 잘 작동했던 경험이 있음
### Xavier initialization
Xavier initialization은 다음과 같은 가정을 기반으로 함

- 활성함수를 선형으로 가정
    - 입력 데이터가 0 근처의 작은 값인 경우에 시그모이드 계열의 활성 함수의 가운데 부분을 지남
    - 이 때 시그모이드 계열의 가운데 부분은 직선에 가까워 선형함수로 가정할 수 있음
- 입력 데이터와 가중치는 다음과 같은 분포의 성질을 가짐
    - 입력 데이터와 가중치는 서로 독립이며, iid (identically and independently distributed)임
    - 입력 데이터와 가중치는 평균이 0인 분포를 따름


Xavier 초기화 식은 다음과 같이 유도 됨
$$
\begin{align*}
\text{Var}{y} &= \text{Var}{\left(\sum_{i=1}^{n} w_ix_i\right)} \\
&= \sum_{i=1}^{n} \text{Var}{(w_ix_i)} \\
&= \sum_{i=1}^{n} E\left[w_i^2 x_i^2\right] - \left(E\left[w_ix_i\right]\right)^2 \\
&= \sum_{i=1}^{n} E\left[w_i^2\right]E\left[x_i^2\right] - \left(E\left[w_i\right]E\left[x_i\right]\right)^2 \\
&= \sum_{i=1}^{n} \left(Var(w_i) + E\left[w_i\right]^2\right)\left(Var(x_i) + E\left[x_i\right]^2\right) - \left(E\left[w_i\right]E\left[x_i\right]\right)^2 \\
&= \sum_{i=1}^{n} \left(Var(w_i) + 0\right)\left(Var(x_i) + 0\right) - \left(0\right)^2 \\
&= \sum_{i=1}^{n} Var(w_i)Var(x_i) \\
&= nVar(w_i)Var(x_i) \\
& \text{가중치 초기화를 통해 입력과 출력의 분산이 같다고 가정} \\
\text{Var}{w_i} &= \frac{1}{n} \\
\end{align*}
$$

위에서 볼 수 있듯이 입력과 출력의 분산을 유지하면서 계산되므로 출력값이 0으로 변하거나 -1, 1로 포화되지 않고 그레디언트 소실 문제를 어느 정도 방지할 수 있다

### He initialization
ReLU 계열의 활성화 함수에서는 음수 구간에서 Xavier 초기화의 가정이 깨지므로 잘 작동하지 않음
HE initialization은 ReLU를 사용했을 때 출력의 분산이 절반으로 줄어들기 때문에 가중치의 분산을 두 배로 키운다. 즉 Xavier 초기화가 가중치의 분산을 $\frac{1}{n}$으로 설정했다면 HE 초기화는 $\frac{2}{n}$으로 설정한다

# 베이즈 정리
베이즈 정리는 조건부 확률을 이용하여 사건의 확률을 계산하는 방법을 제시함
- 베이즈 정리는 다음과 같은 형태를 가짐
$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$
- 여기서 $P(A|B)$는 사건 A가 주어졌을 때 사건 B의 확률을 의미함
    - 사후 확률
- $P(B|A)$는 사건 B가 주어졌을 때 사건 A의 확률을 의미함
    - 우도
- $P(A)$는 사건 A의 확률을 의미함
    - 사전 확률
- $P(B)$는 사건 B의 확률을 의미함
    - 정규화 상수
- 베이즈 정리는 사건 A와 사건 B의 관계를 이용하여 사건 A의 확률을 계산하는 방법을 제시함

B가 일어난 상태에서 A가 발생하는 사건이 희귀한 상황이라고 가정할 때, 이는 관측이 어려운 상황으로도 볼 수 있음

베이지안 정리는 A가 발생했을 때 B가 발생할 확률을 계산하므로써 위 상황을 조금 더 계산하기 쉽게 만들어줌

# 정보량
정보량은 정보의 양을 측정하는 방법으로, 정보의 양이 많을수록 정보량이 큼

정보량은 다음과 같은 식으로 정의됨
$$
I(x) = -\log_2(P(x))
$$
- 여기서 $P(x)$는 사건 x의 확률을 의미함

## 엔트로피
그렇다면 사건 x의 정보량의 평균은 어떻게 계산할까?
$$
H(x) = \sum_{x \in X} P(x) I(x) = -\sum_{x \in X} P(x) \log_2(P(x))
$$

위 식을 보면 알 수 있듯이 정보량의 평균이 엔트로피라는 것을 알 수 있음

## Cross Entropy
Cross Entropy는 두 확률 분포 간의 차이를 측정하는 방법으로, 두 확률 분포가 얼마나 다른지를 나타냄
Cross Entropy는 다음과 같은 식으로 정의됨
$$
CE(p, q) = -\sum_{x \in X} p(x) \log_2(q(x))
$$

여기서 P(x)는 실제 확률 분포, Q(x)는 예측 확률 분포를 의미함
 
크로스 엔트로피를 더 깊게 한 번 이해해보자

$$
\begin{align*}
CE(p, q) &= -\sum_{x \in X} p(x) \log_2(q(x)) \\
&= \sum_{x \in X} p(x) (-\log_2\left(q(x)\right)) \\
&= \sum_{x \in X} p(x) f(q(x)) \\
&= E[f(q(x))] \\
&\geq f(E[(q(x))]) (\text{ Jensen})\\
&= f(\sum_{x \in X} p(x) q(x)) \\
\end{align*}
$$

여기서 $\sum_{x \in X} p(x) q(x)$는 다음과 같음
$$
\sum_{x \in X} p(x) q(x) = P(X_a = Y_a)
$$
즉, 정답일 확률을 의미함

식을 조금 더 정리해보면
$$
\begin{align*}
&CE(p, q) \geq f(P(X_a = Y_a)) = -log_2(P(X_a = Y_a)) \\
&\Leftrightarrow CE(p, q) \geq -log P(X_a = Y_a) \\
&\Leftrightarrow -CE(p, q) \leq log P(X_a = Y_a) \\
&\Leftrightarrow \exp(-CE(p, q)) \leq P(X_a = Y_a) \\

\end{align*}
$$

즉, 크로스 엔트로피 값으로부터 정답일 확률을 계산할 수 있다는 것을 알 수 있음

예를 들어 크로스 엔트로피 값이 1이라면 정답률은 약 0.367이 됨