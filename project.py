import streamlit as st
from sympy import symbols, sympify, lambdify
from sympy.calculus.util import continuous_domain
import numpy as np

x = symbols('x')

# Evaluate function from user input
def get_function(expr_str):
    try:
        expr = sympify(expr_str)
        func = lambdify(x, expr, "math")
        return expr, func
    except Exception:
        return None, None

# Bisection Method
def bisection_method(func, a, b, tol, max_iter):
    steps = []
    if func(a) * func(b) >= 0:
        return "Invalid interval (same signs).", []
    
    for i in range(max_iter):
        c = (a + b) / 2
        steps.append((i+1, a, b, c, func(c)))
        if abs(func(c)) < tol:
            break
        elif func(a) * func(c) < 0:
            b = c
        else:
            a = c
    return c, steps

# Newton-Raphson Method
def newton_raphson(func_expr, func, a, tol, max_iter):
    steps = []
    dfunc = lambdify(x, func_expr.diff(x), "math")
    
    for i in range(max_iter):
        f_val = func(a)
        d_val = dfunc(a)
        if d_val == 0:
            return "Derivative is zero. Cannot proceed.", steps
        a_new = a - f_val / d_val
        steps.append((i+1, a, f_val, d_val, a_new))
        if abs(f_val) < tol:
            break
        a = a_new
    return a, steps

# Streamlit Interface
st.title("Numerical Methods: Bisection & Newton-Raphson")

method = st.selectbox("Choose a Method", ["Bisection Method", "Newton-Raphson Method"])
equation_input = st.text_input("Enter an equation in x (e.g., x**3 - 4*x - 9):")

if equation_input:
    expr, func = get_function(equation_input)
    if not expr:
        st.error("Invalid equation. Please try again.")
    else:
        if method == "Bisection Method":
            a = st.number_input("Enter left bound (a):", value=1.0)
            b = st.number_input("Enter right bound (b):", value=2.0)
        else:
            a = st.number_input("Enter initial guess (a):", value=1.0)

        tol = st.number_input("Enter tolerance (e.g. 0.0001):", value=0.0001)
        max_iter = st.number_input("Maximum number of iterations:", value=50, step=1)

        if st.button("Run"):
            if method == "Bisection Method":
                result, steps = bisection_method(func, a, b, tol, int(max_iter))
                if isinstance(result, str):
                    st.error(result)
                else:
                    st.success(f"Root found: {result}")
                    st.write("Iterations:")
                    st.dataframe(steps, use_container_width=True)
            else:
                result, steps = newton_raphson(expr, func, a, tol, int(max_iter))
                if isinstance(result, str):
                    st.error(result)
                else:
                    st.success(f"Root found: {result}")
                    st.write("Iterations:")
                    st.dataframe(steps, use_container_width=True)
