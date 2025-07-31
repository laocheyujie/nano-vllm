## LLaMA 结构

![LLaMA 结构图](images/LLaMA.png)

注意：Decoder Layer 只有第一个 for 循环的第一个 LayerNorm 输入只有 hidden_states，没有 residual，其他时候 LayerNorm 都是接收 hidden_states 和 residual 两个输入。


## RoPE
原始：

$$ \mathbf{R}^d_{\Theta, m} \mathbf{x} = 
\begin{pmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4 \\
\vdots \\
x_{d-1} \\
x_d
\end{pmatrix}
\otimes
\begin{pmatrix}
\cos m\theta_1 \\
\cos m\theta_1 \\
\cos m\theta_2 \\
\cos m\theta_2 \\
\vdots \\
\cos m\theta_{d/2} \\
\cos m\theta_{d/2}
\end{pmatrix}
+
\begin{pmatrix}
-x_2 \\
x_1 \\
-x_4 \\
x_3 \\
\vdots \\
-x_d \\
x_{d-1}
\end{pmatrix}
\otimes
\begin{pmatrix}
\sin m\theta_1 \\
\sin m\theta_1 \\
\sin m\theta_2 \\
\sin m\theta_2 \\
\vdots \\
\sin m\theta_{d/2} \\
\sin m\theta_{d/2}
\end{pmatrix}
$$

实践中通常将正旋转部分（cos项）和负旋转部分（sin项）分别批量处理，然后再拼接回原始顺序：

$$\mathbf{R}^d_{\Theta, m} \mathbf{x} =
\left[
\begin{pmatrix}
x_1 \\
x_3 \\
\vdots \\
x_{d-1}
\end{pmatrix}
\odot
\begin{pmatrix}
\cos m\theta_1 \\
\cos m\theta_2 \\
\vdots \\
\cos m\theta_{d/2}
\end{pmatrix}
\right]
-
\left[
\begin{pmatrix}
x_2 \\
x_4 \\
\vdots \\
x_d
\end{pmatrix}
\odot
\begin{pmatrix}
\sin m\theta_1 \\
\sin m\theta_2 \\
\vdots \\
\sin m\theta_{d/2}
\end{pmatrix}
\right]
\;\;\bigoplus\;\;
\left[
\begin{pmatrix}
x_2 \\
x_4 \\
\vdots \\
x_d
\end{pmatrix}
\odot
\begin{pmatrix}
\cos m\theta_1 \\
\cos m\theta_2 \\
\vdots \\
\cos m\theta_{d/2}
\end{pmatrix}
\right]
+
\left[
\begin{pmatrix}
x_1 \\
x_3 \\
\vdots \\
x_{d-1}
\end{pmatrix}
\odot
\begin{pmatrix}
\sin m\theta_1 \\
\sin m\theta_2 \\
\vdots \\
\sin m\theta_{d/2}
\end{pmatrix}
\right]
$$