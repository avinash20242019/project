import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sympify, lambdify

x = symbols('x')

def bisection_method(f, a, b, iterations):
    results = []
    if f(a) * f(b) > 0:
        return None, ["f(a) and f(b) should have opposite signs."]
    
    for i in range(iterations):
        c = (a + b) / 2
        fc = f(c)
        results.append((i + 1, a, b, c, fc))
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    return c, results

def newton_raphson_method(f, df, x0, iterations):
    results = []
    for i in range(iterations):
        fx = f(x0)
        dfx = df(x0)
        if dfx == 0:
            return None, ["Zero derivative. Cannot proceed."]
        x1 = x0 - fx / dfx
        results.append((i + 1, x0, fx, x1))
        x0 = x1
    return x1, results

def plot_function(f, root=None, title="Function Plot"):
    xs = np.linspace(-10, 10, 400)
    ys = [f(val) for val in xs]
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, label='f(x)')
    plt.axhline(0, color='gray', linestyle='--')
    if root is not None:
        plt.plot(root, f(root), 'ro', label='Root')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    st.pyplot(plt)

st.title("Numerical Methods: Bisection and Newton-Raphson")

method = st.selectbox("Select Method", ["Bisection Method", "Newton-Raphson Method"])
expr_input = st.text_input("Enter function f(x):", "x**3 - x - 2")
iterations = st.number_input("Number of Iterations", min_value=1, value=10, step=1)

try:
    expr = sympify(expr_input)
    f = lambdify(x, expr, "numpy")
    df_expr = expr.diff(x)
    df = lambdify(x, df_expr, "numpy")

    if method == "Bisection Method":
        a = st.number_input("Enter interval start (a):", value=1.0)
        b = st.number_input("Enter interval end (b):", value=2.0)
        if st.button("Run Bisection Method"):
            root, result = bisection_method(f, a, b, iterations)
            if isinstance(result, list) and isinstance(result[0], str):
                st.error(result[0])
            else:
                st.success(f"Approximated root after {iterations} iterations: {root:.6f}")
                st.markdown("### Iteration Results")
                for i, a_i, b_i, c_i, fc in result:
                    st.markdown(f"<span style='color:blue'>Iteration {i}:</span> a={a_i:.6f}, b={b_i:.6f}, c={c_i:.6f}, f(c)={fc:.6e}", unsafe_allow_html=True)
                plot_function(f, root, "Bisection Method Result")

    elif method == "Newton-Raphson Method":
        x0 = st.number_input("Enter initial guess (x₀):", value=1.5)
        if st.button("Run Newton-Raphson Method"):
            root, result = newton_raphson_method(f, df, x0, iterations)
            if isinstance(result, list) and isinstance(result[0], str):
                st.error(result[0])
            else:
                st.success(f"Approximated root after {iterations} iterations: {root:.6f}")
                st.markdown("### Iteration Results")
                for i, x_i, fx, x_next in result:
                    st.markdown(f"<span style='color:green'>Iteration {i}:</span> x₀={x_i:.6f}, f(x₀)={fx:.6e}, x₁={x_next:.6f}", unsafe_allow_html=True)
                plot_function(f, root, "Newton-Raphson Method Result")

except Exception as e:
    st.error(f"Error: {e}")

