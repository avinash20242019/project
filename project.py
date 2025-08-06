import streamlit as st
from sympy import symbols, sympify, lambdify, diff

x = symbols('x')

def bisection_method(func_expr, a, b, tol, max_iter):
    f = lambdify(x, sympify(func_expr), "math")
    results = []
    if f(a) * f(b) >= 0:
        return None, ["Function has same signs at a and b. Cannot apply Bisection Method."]
    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        results.append((i, a, b, c, fc))
        if abs(fc) < tol or abs(b - a) < tol:
            break
        if f(a) * fc < 0:
            b = c
        else:
            a = c
    return c, results

def newton_raphson_method(func_expr, x0, tol, max_iter):
    f_expr = sympify(func_expr)
    f = lambdify(x, f_expr, "math")
    f_prime = lambdify(x, diff(f_expr, x), "math")
    results = []
    for i in range(1, max_iter + 1):
        f_x0 = f(x0)
        f_prime_x0 = f_prime(x0)
        if f_prime_x0 == 0:
            return None, ["Zero derivative. Cannot continue."]
        x1 = x0 - f_x0 / f_prime_x0
        results.append((i, x0, f_x0, x1))
        if abs(x1 - x0) < tol:
            break
        x0 = x1
    return x1, results

st.set_page_config(page_title="Numerical Methods App", layout="centered")
st.title("Numerical Methods App")

method = st.sidebar.selectbox("Choose Method", ["Bisection Method", "Newton-Raphson Method"])

with st.form(key='form'):
    func_input = st.text_input("Enter function f(x):", "x**3 - x - 2")
    tol = st.number_input("Tolerance:", min_value=1e-10, value=0.0001, format="%.5f")
    max_iter = st.number_input("Maximum Iterations:", min_value=1, value=20)
    if method == "Bisection Method":
        a = st.number_input("Enter interval start (a):", value=1.0)
        b = st.number_input("Enter interval end (b):", value=2.0)
    else:
        x0 = st.number_input("Enter initial guess (x₀):", value=1.5)
    submit = st.form_submit_button("Run")

if submit:
    st.markdown(f"### Method: {method}")
    st.markdown(f"**Function:** `{func_input}`")
    if method == "Bisection Method":
        root, steps = bisection_method(func_input, a, b, tol, max_iter)
        if root is None:
            for err in steps:
                st.error(err)
        else:
            for i, a_, b_, c_, fc in steps:
                color = "green" if abs(fc) < tol else "black"
                st.markdown(
                    f"<span style='color:{color}'>Iteration {i}: a = {a_:.6f}, b = {b_:.6f}, c = {c_:.6f}, f(c) = {fc:.6f}</span>",
                    unsafe_allow_html=True
                )
            st.success(f"Approximated root: {root:.6f}")
    else:
        root, steps = newton_raphson_method(func_input, x0, tol, max_iter)
        if root is None:
            for err in steps:
                st.error(err)
        else:
            for i, x0_, fx0, x1 in steps:
                color = "green" if abs(x1 - x0_) < tol else "black"
                st.markdown(
                    f"<span style='color:{color}'>Iteration {i}: x₀ = {x0_:.6f}, f(x₀) = {fx0:.6f}, x₁ = {x1:.6f}</span>",

