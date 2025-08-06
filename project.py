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
            return None, ["Zero derivat]()

