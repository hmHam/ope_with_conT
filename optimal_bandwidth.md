### Epanechnikov kernelの$\kappa_2(K)$および$R(K)$

* この論文ではEpanechnikov kernelは$K(u) = \frac{3}{4} (1 - u^2) \mathbb{I}\{\|u \| \le 1\}$となっている。
* なので、second moment$\kappa_2(K) = \int u^2 K(u) du$は、$\int u^2 K(u) du = \left[\frac{3}{4}(x^5/5 - 2x^3/3 + x\right]_{-1}^{1} = \frac{3}{4}\frac{16}{15} = \frac{4}{5}$
* さらに、roughness$R(K) = \int K(u)^2du$は、$\int K(u)^2 du = \frac{9}{16} \left[ x^3/3 - x^5/5 \right]_{-1}^1 = \frac{9}{16} \frac{4}{15} = \frac{3}{20}$
* よって、実験のoptimal bandwidthの算出にはこの数値を利用する。
* 計算には、[このツール](https://ja.wolframalpha.com/input/?i=%E7%A9%8D%E5%88%86%E8%A8%88%E7%AE%97)を利用した。


### Optimal Bandwidth
* section3.2のTheorem2を利用する。
$
h^{*} = \left(
    \frac{
        R(K) \mathbb{E}\left[\mathbb{E}[Y^2|\tau(X), X]/f_{T|X}(\tau(X), X)\right]
    }{
        4 \left(
            \mathbb{E} \left[ \int \frac{y_i}{2} \frac{\partial}{\partial T^2} f_{Y|T, x}(y_i, \tau(x_i)) \kappa_2(K) dy \right]
        \right)^2 n
    }
\right)^{\frac{1}{5}}
$

* section5.1によれば、2次の微分係数や積分は数値微分, 数値積分を用いる。ここでは、numpy.gradient(y, dx), scipy.integrate.simps(y, x)を利用する。
    * [scipy.integrate.simps](https://qiita.com/sci_Haru/items/09279cf81b9b073afa1d)

* $E[Y^2|\pi(x), X]$は生成したデータのうち、$T = \pi(X)$が成り立つサンプルの2

* $f_{Y|T, x}(y_i, \pi(x_i))$, $f_{T|X}(\pi(X), X)$が謎？