import streamlit as st
from sympy import symbols, sympify, lambdify, diff

x = symbols('x')

def bisection_method(expr, a, b, tol, max_iter):
    f = lambdify(x, expr, "math")
    steps = []
    if f(a) * f(b) >= 0:
        return None, ["Invalid interval"]
    for i in range(1, max_iter + 1):
        c = (a + b) / 2.0
        fc = f(c)
        steps.append((i, a, b, c, fc))
        if abs(fc) < tol or abs(b - a) < tol:
            return c, steps
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    return c, steps

def newton_raphson_method(expr, x0, tol, max_iter):
    f = lambdify(x, expr, "math")
    f_prime = lambdify(x, diff(expr, x), "math")
    steps = []
    for i in range(1, max_iter + 1):
        fx = f(x0)
        dfx = f_prime(x0)
        if dfx == 0:
            return None, ["Zero derivative"]
        x1 = x0 - fx / dfx
        steps.append((i, x0, fx, x1))
        if abs(x1 - x0) < tol:
            return x1, steps
        x0 = x1
    return x1, steps

st.set_page_config(page_title="Numerical Methods", layout="centered")
st.title("Numerical Methods App")

method = st.selectbox("Choose a method", ["Bisection Method", "Newton-Raphson Method"])

expr_str = st.text_input("Enter the function f(x):", "x**3 - x - 2")
try:
    expr = sympify(expr_str)
except:
    st.error("Invalid function expression")
    st.stop()

tol = st.number_input("Tolerance", min_value=1e-10, max_value=1.0, value=1e-6, step=1e-6, format="%.10f")
max_iter = st.number_input("Maximum Iterations", min_value=1, max_value=1000, value=100)

def style_text(text, color):
    return f"<span style='color:{color}; font-weight:bold;'>{text}</span>"

if method == "Bisection Method":
    a = st.number_input("Enter the lower bound a:", value=1.0)
    b = st.number_input("Enter the upper bound b:", value=2.0)
    if st.button("Compute"):
        result, steps = bisection_method(expr, a, b, tol, max_iter)
        if isinstance(steps[0], str):
            st.error(steps[0])
        else:
            for i, a_, b_, c_, fc in steps:
                if abs(fc) < tol:
                    color = "green"
                elif abs(fc) < 10 * tol:
                    color = "orange"
                else:
                    color = "blue"
                st.markdown(
                    style_text(f"Iteration {i}: a = {a_:.6f}, b = {b_:.6f}, c = {c_:.6f}, f(c) = {fc:.6f}", color),
                    unsafe_allow_html=True
                )
            st.success(f"Approximated root: {result:.6f}")

elif method == "Newton-Raphson Method":
    x0 = st.number_input("Enter initial guess x₀:", value=1.5)
    if st.button("Compute"):
        result, steps = newton_raphson_method(expr, x0, tol, max_iter)
        if isinstance(steps[0], str):
            st.error(steps[0])
        else:
            for i, x0_, fx0, x1 in steps:
                if abs(x1 - x0_) < tol:
                    color = "green"
                elif abs(x1 - x0_) < 10 * tol:
                    color = "orange"
                else:
                    color = "blue"
                st.markdown(
                    style_text(f"Iteration {i}: x₀ = {x0_:.6f}, f(x₀) = {fx0:.6f}, x₁ = {x1:.6f}", color),
                    unsafe_allow_html=True
                )
            st.success(f"Approximated root: {result:.6f}")
