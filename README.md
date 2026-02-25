🧮 Numerical Methods Solver
Bisection & Newton–Raphson Method (Streamlit App)

🔗 Live App:
https://project-sr8csiuzufqyirs9hhfztt.streamlit.app/

📌 Overview

This web application implements two important root-finding numerical methods:

Bisection Method

Newton–Raphson Method

Choose the numerical method

Set number of iterations

Provide initial guess (for Newton-Raphson)

Instantly compute the approximate root

It is built using Python and Streamlit to provide an interactive and user-friendly interface.

🚀 Features

✅ Select between Bisection and Newton–Raphson methods
✅ Accepts custom mathematical functions (e.g., x**3 - x - 2)
✅ User-defined number of iterations
✅ Initial guess input for Newton–Raphson
✅ Displays final approximated root
✅ Clean dark-themed UI

🧠 Mathematical Background
🔹 Bisection Method


🛠️ Tech Stack

Python

Streamlit

SymPy (if used for derivatives)

NumPy (if used)

💻 How to Run Locally
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run app.py
📂 Project Structure
📦 Numerical-Methods-App
 ┣ 📜 app.py
 ┣ 📜 requirements.txt
 ┣ 📜 README.md
📊 Example

Input:

f(x) = x^3 - x - 2
Initial guess = 1.5
Iterations = 10

Output:

Approximated root after 10 iterations: 1.521380
🎯 Learning Purpose

This project was built to:

Understand numerical root-finding techniques

Visualize iterative convergence

Apply mathematical concepts in a real interactive application

Practice Python + Streamlit deployment

🌍 Deployment

Deployed using Streamlit Community Cloud.

📜 License

This project is for educational purposes.
