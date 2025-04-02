# 스칼라와 벡터
## 스칼라와 벡터의 차이
스칼라: 크기
벡터: 방향, 크기

# 벡터와 행렬의 표현
## 벡터의 표현
$$\mathbf{x} = (x_1, x_2, ... x_n)$$

위 벡터의 차원은 n
$$x \in \mathbb{R}^n$$

2차원 벡터 그려보기, 3차원 벡터 그려보기
## 행렬의 표현
행렬은 벡터로 구성되어 있다고 할 수 있음

이후에 이야기할 텐서의 개념임

$$
\mathbf{A} = 
\begin{bmatrix}
   a_{11} & a_{12} & a_{13}  \\
   a_{21} & a_{22} & a_{23}  \\
   a_{31} & a_{32} & a_{33}  \\
\end{bmatrix}
 =
 \begin{bmatrix}
   \mathbf{a_{1}} & \mathbf{a_{2}} & \mathbf{a_{3}}  \\
\end{bmatrix}$$

행렬의 차원은 각 벡터의 차원의 합과 같음

위의 예시의 경우에는
$$ \mathbf{A} \in \mathbb{R}^{3n} $$


# 내적과 외적

## 외적
### 외적의 계산

$$ \mathbf{u} \times \mathbf{v} = 
\begin{vmatrix}
\mathbf{e_1} & \mathbf{e_2} & \mathbf{e_3} \\
u_1 & u_2 & u_3 \\
v_1 & v_2 & v_3 \\
\end{vmatrix} $$

### 외적의 기하학적 의미
외적의 크기는 아래 식으로 표현할 수 있음
$$ \mathbf{u} \times \mathbf{v} = \lvert \mathbf{u}\rvert \lvert \mathbf{v} \rvert \sin{\theta}\ \mathbf{n}$$

여기서 $\mathbf{n}$은 $\mathbf{u}$와 $\mathbf{v}$의 수직인 단위 벡터를 의미함

![](attachments/Pasted%20image%2020250401190319.png)

또한, 그 크기는 $\mathbf{u}$와 $\mathbf{v}$로 이루어진 평행사변형의 넓이와 같음
(2차원 상에서 그려보기)

(문제) $\mathbf{i} \cdot (\mathbf{j} \times \mathbf{k})$의 결과를 계산하고 의미를 생각해보기

## 내적
### 내적의 계산

$$ \mathbf{u} \cdot \mathbf{v} = u_1 \times v_1 + u_2 \times v_2 + ... + u_n \times v_n $$

위 수식을 보면 알겠지만 두 벡터의 차원이 같을 때에만 성립함

### 내적의 기하학적 의미
내적은 아래 식으로도 표현할 수 있음
$$ \mathbf{u} \cdot \mathbf{v}  = \lvert \mathbf{u}\rvert \lvert \mathbf{v} \rvert \cos{\theta}
$$

(문제) 내적의 계산과 내적의 기하학적 의미에 대해 두 식이 같음을 증명하면서 $\theta$가 어디의 각도를 의미하는지 생각해보기 (제2코사인함수 참고)

2차원 상에서 그려보기


## 정사영과 코사인 유사도
### 정사영
正射影 (바를 정, 쏠 사, 비출 영 (그림자 영))
정사영은 그림자를 의미함을 생각하고 기하적으로 바라볼 것

(두 벡터를 그리고) 벡터 $\mathbf{u}$를 $\mathbf{v}$에 정사영한 벡터를 $\mathbf{u'}$라고 하자
$\mathbf{u'}$는 어떻게 그릴 수 있을까?



$$ \mathbf{u'} = \lvert \mathbf{u} \rvert \cos{\theta}\  \frac{\mathbf{v}}{\lvert \mathbf{v} \rvert} = \frac{\mathbf{u} \cdot \mathbf{v}}{\lvert \mathbf{v} \rvert^2} \mathbf{v}$$

### 코사인 유사도
[내적의 기하학적 의미](#내적의%20기하학적%20의미)를 생각하며 

우리는 왜 cos유사도를 사용하고 있었을까
(컴퓨팅에 있어 계산 편의성 언급)

SIMD와 GPU를 활용한 벡터 연산

# 텐서와 행렬 연산
## 텐서의 정의
좌표변환하에서 특정한 변환법칙을 따르는 양이라고할 수 있음

수학적으로는 기하학에서의 텐서와 대수학에서의 텐서가 조금 다르나, 공학적인 관점에서는 크게 다르지 않다고 생각됨. 더 자세하게 알고 있는 사람이 있다면 추가 설명 부탁

보통 행렬이 쌓여있는 형태로 표현하며, Computer Vision에서는 흔히 표현되는 $N \times C \times H \times W$의 형태의 텐서를 생각할 수 있음
(여기서 $N$은 배치사이즈, $C$는 채널, $H$는 높이, $W$는 너비를 의미함)

본인의 경우에는 2차원이 넘는 행렬의 경우를 보통 텐서라고 부른다고 생각하고 있음

## 행렬 연산
### 행렬의 덧셈, 뺄셈
행렬의 덧셈과 뺄셈은 각 원소끼리 더하거나 빼는 것
$$ \mathbf{A} =
\begin{bmatrix}
   a_{11} & a_{12} & a_{13}  \\
   a_{21} & a_{22} & a_{23}  \\
   a_{31} & a_{32} & a_{33}  \\
\end{bmatrix}
\pm
\mathbf{B} =
\begin{bmatrix}
   b_{11} & b_{12} & b_{13}  \\
   b_{21} & b_{22} & b_{23}  \\
   b_{31} & b_{32} & b_{33}  \\
\end{bmatrix}
=
\begin{bmatrix}
   a_{11} \pm b_{11} & a_{12} \pm b_{12} & a_{13} \pm b_{13}  \\
   a_{21} \pm b_{21} & a_{22} \pm b_{22} & a_{23} \pm b_{23}  \\
   a_{31} \pm b_{31} & a_{32} \pm b_{32} & a_{33} \pm b_{33}  \\
\end{bmatrix}
$$

### 행렬의 곱셈
#### 상수배
행렬의 각 원소에 상수를 곱하는 것
$$ \mathbf{A} =
\begin{bmatrix}
   a_{11} & a_{12} & a_{13}  \\
   a_{21} & a_{22} & a_{23}  \\
   a_{31} & a_{32} & a_{33}  \\
\end{bmatrix}
\cdot k =
\begin{bmatrix}
   k \cdot a_{11} & k \cdot a_{12} & k \cdot a_{13}  \\
   k \cdot a_{21} & k \cdot a_{22} & k \cdot a_{23}  \\
   k \cdot a_{31} & k \cdot a_{32} & k \cdot a_{33}  \\
\end{bmatrix}
$$

#### 아다마르 곱
행렬의 각 원소끼리 곱하는 것
$$ \mathbf{A} =
\begin{bmatrix}
   a_{11} & a_{12} & a_{13}  \\
   a_{21} & a_{22} & a_{23}  \\
   a_{31} & a_{32} & a_{33}  \\
\end{bmatrix}
\odot
\mathbf{B} =
\begin{bmatrix}
   b_{11} & b_{12} & b_{13}  \\
   b_{21} & b_{22} & b_{23}  \\
   b_{31} & b_{32} & b_{33}  \\
\end{bmatrix}
=
\begin{bmatrix}
   a_{11} \cdot b_{11} & a_{12} \cdot b_{12} & a_{13} \cdot b_{13}  \\
   a_{21} \cdot b_{21} & a_{22} \cdot b_{22} & a_{23} \cdot b_{23}  \\
   a_{31} \cdot b_{31} & a_{32} \cdot b_{32} & a_{33} \cdot b_{33}  \\
\end{bmatrix}
$$

#### 행렬 곱
행렬의 곱은 행렬의 각 원소를 곱하는 것이 아니라, 앞의 행렬의 행을 벡터로 생각하고 뒤의 행렬의 열을 벡터로 생각하여 내적을 하는 것

$$ \mathbf{A} =
\begin{bmatrix}
   \mathbf{a_{1}} \\
   \mathbf{a_{2}} \\
   \mathbf{a_{3}} \\
\end{bmatrix}
\cdot
\mathbf{B} =
\begin{bmatrix}
   \mathbf{b_{1}} & \mathbf{b_{2}} & \mathbf{b_{3}}  \\
\end{bmatrix}
=
\begin{bmatrix}
   \mathbf{a_{1}} \cdot \mathbf{b_{1}} & \mathbf{a_{1}} \cdot \mathbf{b_{2}} & \mathbf{a_{1}} \cdot \mathbf{b_{3}}  \\
   \mathbf{a_{2}} \cdot \mathbf{b_{1}} & \mathbf{a_{2}} \cdot \mathbf{b_{2}} & \mathbf{a_{2}} \cdot \mathbf{b_{3}}  \\
   \mathbf{a_{3}} \cdot \mathbf{b_{1}} & \mathbf{a_{3}} \cdot \mathbf{b_{2}} & \mathbf{a_{3}} \cdot \mathbf{b_{3}}  \\
\end{bmatrix}
$$

내적을 한다는 것? => 유사도를 본다는 것.

##### Linear Projection에 대해서
흔히 Linear Projection이라고 하는데, 이는 행렬과 벡터의 곱을 내적으로 해석하며 이해할 수 있음

예를 들어 $C \times H \times W$의 데이터를 $1 \times H \times W$의 데이터로 Linear Projection한다고 생각하면, 이는 다음과 같은 수식으로 표현됨
$$ \mathbf{W} \cdot \mathbf{x} = \mathbf{y}$$
여기서 $\mathbf{W}$는 $1 \times C$의 행렬, $\mathbf{X}$는 $C \times H \times W$의 텐서, $\mathbf{y}$는 $1 \times H \times W$의 텐서라고 볼 수 있음




추후에 공분산행렬에서도 다시 나온다는 언급

---

행렬의 곱은 행렬 간의 연산보다 행렬과 벡터간의 연산에서 더 많은 인사이트를 제공

행렬과 벡터의 곱은 여러가지 해석이 가능함

##### 열벡터의 선형결합으로 해석
여기서 잠깐! 선형 결합이란?

각 항에 상수를 곱하고 결과를 더함으로써 일련의 항으로 구성된 표현식

하지만 조건이 하나 있는데, 선형결합의 출력이 입력과 같은 공간을 가져야함.
이를 선형결합에 대해 닫혀있다라고 함. 어떤 연산에 대해 닫혀있다라는 표현은 입력과 결과의 공간이 같다는 뜻.


$$
\mathbf{A} =
\begin{bmatrix}
   a_{11} & a_{12} & a_{13}  \\
   a_{21} & a_{22} & a_{23}  \\
   a_{31} & a_{32} & a_{33}  \\
\end{bmatrix}
*
\mathbf{v} =
\begin{bmatrix}
   v_1 \\
   v_2 \\
   v_3 \\
\end{bmatrix}
=
\begin{bmatrix}
   a_{11} \cdot v_1 + a_{12} \cdot v_2 + a_{13} \cdot v_3  \\
   a_{21} \cdot v_1 + a_{22} \cdot v_2 + a_{23} \cdot v_3  \\
   a_{31} \cdot v_1 + a_{32} \cdot v_2 + a_{33} \cdot v_3  \\
\end{bmatrix}
=
\begin{bmatrix}
   \mathbf{a_{1}} \cdot v_1 + \mathbf{a_{2}} \cdot v_2 + \mathbf{a_{3}} \cdot v_3  \\
\end{bmatrix}
$$

위 식에서 $v_1, v_2, v_3$를 상수로 보면, 행렬 $\mathbf{A}$의 열벡터 $\mathbf{a_{1}}, \mathbf{a_{2}}, \mathbf{a_{3}}$의 선형결합으로 볼 수 있음

##### 선형 변환으로 해석
행렬 $\mathbf{A}$를 선형 변환으로 해석할 수 있음

$$
\mathbf{A} =
\begin{bmatrix}
   a_{11} & a_{12} & a_{13}  \\
   a_{21} & a_{22} & a_{23}  \\
   a_{31} & a_{32} & a_{33}  \\
\end{bmatrix}
\cdot
\begin{bmatrix}
   x_1 \\
   x_2 \\
   x_3 \\
\end{bmatrix}
$$
에서 $\mathbf{A}$는 선형 변환에 사용되는 행렬으로 볼 수 있음

추후에 선형변환에 대해 더 자세하게 설명

# 벡터와 행렬의 미분
## 스칼라를 벡터로 미분하는 경우
입력이 벡터이고, 스칼라가 벡터인 함수를 생각해보자.

$$ f: \mathbb{R}^n \to \mathbb{R} $$
$$ f(\mathbf{x}) = y $$

수식으로만 보면 이해가 잘 되지 않을 수 있다. 예시로 간단한 함수 하나를 살펴보자
$$ f(\mathbf{x}) = f(x_1, x_2) = x_1^2 + x_2^2 $$

![](attachments/Pasted%20image%2020250402104117.png)

이러한 함수도 벡터를 입력으로 받아서 스칼라를 반환하는 예시라고 볼 수 있음

이 함수를 미분하기 위해서는 어떻게 해야할까?

미분은 변화율을 의미함. 2차원의 함수에서는 그 변화율이 명확했지만 3차원, 4차원, n차원으로 확장되면 그 의미가 모호해짐.

그렇기 때문에, 우리는 편미분을 사용해야 함.

편미분의 편은 마치 마늘을 편 썬다는 것처럼 얇게 잘라낸다고 생각할 수 있음.

그렇게 얇게 잘라내면, 마치 2차원의 함수처럼 볼 수 있기 때문임.

편미분을 식으로 나타내면

$$ \Delta f = \frac{\partial f}{\partial \mathbf{x}} = 
\begin{bmatrix}
   \frac{\partial f}{\partial x_1} \\
   \frac{\partial f}{\partial x_2} \\
   \vdots \\
   \frac{\partial f}{\partial x_n} \\
\end{bmatrix}
$$

각 원소는 해당 기저벡터에서의 변화율을 의미함

## 벡터를 스칼라로 미분하는 경우

$$ f: \mathbb{R} \to \mathbb{R}^n $$
$$ \frac{\partial f}{\partial x} =
\begin{bmatrix}
   \frac{\partial f}{\partial x} &
   \frac{\partial f}{\partial x} &
   \cdots &
   \frac{\partial f}{\partial x_n}
\end{bmatrix}
$$

## 벡터를 벡터로 미분하는 경우

$$ f: \mathbb{R}^n \to \mathbb{R}^m $$
$$ \frac{\partial f}{\partial \mathbf{x}} =
\begin{bmatrix}
   \frac{\partial f}{\partial x_1} \\
   \frac{\partial f}{\partial x_2} \\
   \vdots \\
   \frac{\partial f}{\partial x_n} \\
\end{bmatrix}
=
\begin{bmatrix}
   \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
   \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
   \vdots & \vdots & \ddots & \vdots \\
   \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \\
\end{bmatrix}
$$


## 스칼라를 행렬로 미분
$$ f: \mathbb{R}^{m \times n} \to \mathbb{R} $$
$$ \frac{\partial f}{\partial \mathbf{X}} =
\begin{bmatrix}
   \frac{\partial f}{\partial x_{11}} & \frac{\partial f}{\partial x_{12}} & \cdots & \frac{\partial f}{\partial x_{1n}} \\
   \frac{\partial f}{\partial x_{21}} & \frac{\partial f}{\partial x_{22}} & \cdots & \frac{\partial f}{\partial x_{2n}} \\
   \vdots & \vdots & \ddots & \vdots \\
   \frac{\partial f}{\partial x_{m1}} & \frac{\partial f}{\partial x_{m2}} & \cdots & \frac{\partial f}{\partial x_{mn}} \\
\end{bmatrix}