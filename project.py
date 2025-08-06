import streamlit as st
from sympy import symbols, sympify, lambdify, diff
import numpy as np

st.set_page_config(page_title="Numerical Methods App", layout="centered")
st.title("Numerical Methods Solver")

st.header("Input Parameters")
equation = st.text_input("Enter the function f(x):", value="x**3 - x - 2")
method = st.selectbox("Choose a method:", ["Bisection Method", "Newton-Raphson Method"])

if method == "Bisection Method":
    a = st.number_input("Enter interval start (a):", value=1.0)
    b = st.number_input("Enter interval end (b):", value=2.0)
else:
    x0 = st.number_input("Enter initial guess (x₀):", value=1.5)

tolerance = st.number_input("Enter tolerance:", value=1e-6, format="%.1e")
max_iter = st.number_input("Enter max iterations:", value=20, step=1)

if st.button("Run Method"):

    x = symbols('x')
    try:
        f_expr = sympify(equation)
        f = lambdify(x, f_expr, modules=["numpy"])
        f_prime = None
        if method == "Newton-Raphson Method":
            f_prime_expr = diff(f_expr, x)
            f_prime = lambdify(x, f_prime_expr, modules=["numpy"])
    except Exception as e:
        st.error(f"Invalid function. Error: {e}")
        st.stop()

    def bisection_method(f, a, b, tol, max_iter):
        steps = []
        if f(a) * f(b) >= 0:
            steps.append("Invalid interval: f(a) and f(b) must have opposite signs.")
            return None, steps
        for i in range(max_iter):
            c = (a + b) / 2
            steps.append(f"Iteration {i+1}: a={a:.6f}, b={b:.6f}, c={c:.6f}, f(c)={f(c):.6e}")
            if abs(f(c)) < tol or abs(b - a) < tol:
                steps.append(f"Converged after {i+1} iterations.")
                break
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c
        else:
            steps.append("Maximum iterations reached without convergence.")
        return c, steps

    def newton_raphson_method(f, f_prime, x0, tol, max_iter):
        steps = []
        for i in range(max_iter):
            f_x0 = f(x0)
            f_prime_x0 = f_prime(x0)
            if f_prime_x0 == 0:
                steps.append("Derivative is zero. Division by zero.")
                return None, steps
            x1 = x0 - f_x0 / f_prime_x0
            steps.append(f"Iteration {i+1}: x₀={x0:.6f}, f(x₀)={f_x0:.6e}, x₁={x1:.6f}")
            if abs(x1 - x0) < tol:
                steps.append(f"Converged after {i+1} iterations.")
                break
            x0 = x1
        else:
            steps.append("Maximum iterations reached without convergence.")
        return x1, steps

    if method == "Bisection Method":
        root, steps = bisection_method(f, a, b, tolerance, int(max_iter))
    else:
        root, steps = newton_raphson_method(f, f_prime, x0, tolerance, int(max_iter))

    st.header("Results")
    for step in steps:
        st.text(step)

    if root is not None:
        st.success(f"Approximated root: {root:.6f}")

