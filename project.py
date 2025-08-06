import streamlit as st
from sympy import symbols, sympify, lambdify, diff
import numpy as np

x = symbols('x')

# --------------------------
# Numerical Method Functions
# --------------------------

def bisection_method(func_expr, a, b, tol, max_iter):
    f = lambdify(x, func_expr, 'numpy')
    steps = []
    if f(a) * f(b) >= 0:
        return None, ["Bisection method fails. f(a) and f(b) must have opposite signs."]
    
    for i in range(max_iter):
        c = (a + b) / 2
        steps.append(f"Iteration {i+1}: a = {a:.6f}, b = {b:.6f}, c = {c:.6f}, f(c) = {f(c):.6f}")
        if abs(f(c)) < tol or (b - a) / 2 < tol:
            return c, steps
        if f(c) * f(a) < 0:
            b = c
        else:
            a = c
    return c, steps

def newton_raphson_method(func_expr, x0, tol, max_iter):
    f = lambdify(x, func_expr, 'numpy')
    df = lambdify(x, diff(func_expr, x), 'numpy')
    steps = []

    for i in range(max_iter):
        fx = f(x0)
        dfx = df(x0)
        if dfx == 0:
            return None, [f"Derivative zero at x = {x0}. Cannot continue."]
        x1 = x0 - fx / dfx
        steps.append(f"Iteration {i+1}: x = {x0:.6f}, f(x) = {fx:.6f}, f'(x) = {dfx:.6f}, next x = {x1:.6f}")
        if abs(x1 - x0) < tol:
            return x1, steps
        x0 = x1
    return x1, steps

# --------------------------
# Streamlit UI
# --------------------------

st.set_page_config(page_title="Numerical Methods App", layout="centered")
st.title("📐 Numerical Methods Solver")
st.markdown("Solve nonlinear equations using **Bisection** or **Newton-Raphson** method.")

# Input common to both methods
method = st.selectbox("Choose a method", ["Bisection Method", "Newton-Raphson Method"])
equation = st.text_input("Enter the equation f(x) =", value="x**3 - x - 2")
tol = st.number_input("Tolerance", value=1e-6, format="%.1e")
max_iter = st.number_input("Maximum Iterations", value=20, step=1)

try:
    expr = sympify(equation)
    st.latex(f"f(x) = {expr}")
except Exception as e:
    st.error("Invalid function. Please enter a valid Python expression like `x**3 - x - 2`.")
    st.stop()

# Method-specific input
if method == "Bisection Method":
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("Enter lower bound a", value=1.0)
    with col2:
        b = st.number_input("Enter upper bound b", value=2.0)
    if st.button("Solve"):
        root, steps = bisection_method(expr, a, b, tol, int(max_iter))
        if root is not None:
            st.success(f"Estimated Root: {root:.6f}")
        for step in steps:
            st.write(step)

elif method == "Newton-Raphson Method":
    x0 = st.number_input("Enter initial guess x₀", value=1.5)
    if st.button("Solve"):
        root, steps = newton_raphson_method(expr, x0, tol, int(max_iter))
        if root is not None:
            st.success(f"Estimated Root: {root:.6f}")
        for step in steps:
            st.write(step)

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit by [Avinash](https://github.com/avinash20242019)")

