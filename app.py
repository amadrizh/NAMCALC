from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sympy import symbols, sympify, diff, lambdify
import re


app = Flask(__name__)

x = symbols('x')
plot_url = None  # Reset before defining a new one




def preprocess_function(func_str):
    func_str = func_str.replace('^', '**')
    func_str = func_str.replace("e", "exp(1)")
    func_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', func_str)
    return func_str



def newton_method(func_str, x0, iterations, decimal_places):
    f = sympify(func_str)
    f_prime = diff(f, x)  # Derivative of the function
    f_np = lambdify(x, f, "numpy")
    f_prime_np = lambdify(x, f_prime, "numpy")

    x_vals = [x0]
    results = []

    for i in range(iterations):
        f_xi = f_np(x_vals[-1])
        f_prime_xi = f_prime_np(x_vals[-1])

        if f_prime_xi == 0:
            break  # Avoid division by zero

        x_new = x_vals[-1] - f_xi / f_prime_xi
        x_vals.append(x_new)

        results.append([
            i + 1,
            round(x_vals[-2], decimal_places),
            round(f_xi, decimal_places),
            round(f_prime_xi, decimal_places),
            round(x_new, decimal_places)
        ])

    return results, x_vals, f_np, "newton"


def bisection_method(func_str, a, b, iterations, decimal_places):
    f = sympify(func_str)
    f_np = lambdify(x, f, "numpy")

    if f_np(a) * f_np(b) > 0:
        raise ValueError("Function must have opposite signs at endpoints a and b.")

    results = []
    x_vals = []

    for i in range(iterations):
        c = (a + b) / 2  # Midpoint
        f_a = f_np(a)
        f_b = f_np(b)
        f_c = f_np(c)

        results.append([
            i + 1,
            round(a, decimal_places),
            round(b, decimal_places),
            round(c, decimal_places),
            round(f_c, decimal_places)
        ])

        x_vals.append((a, b))  # Store the interval for plotting

        if f_c == 0:
            break
        elif f_a * f_c < 0:
            b = c
        else:
            a = c

    return results, x_vals, f_np, "bisection"


def secant_method(func_str, x0, x1, iterations, decimal_places):
    f = sympify(func_str)
    f_np = lambdify(x, f, "numpy")

    x_vals = [x0, x1]
    results = []

    for i in range(iterations):
        f_x0 = f_np(x_vals[-2])
        f_x1 = f_np(x_vals[-1])

        if f_x1 - f_x0 == 0:
            break  # Avoid division by zero

        x_new = x_vals[-1] - f_x1 * (x_vals[-1] - x_vals[-2]) / (f_x1 - f_x0)
        x_vals.append(x_new)

        results.append([
            i + 1,
            round(x_vals[-3], decimal_places),
            round(x_vals[-2], decimal_places),
            round(f_x0, decimal_places),
            round(f_x1, decimal_places),
            round(x_new, decimal_places)
        ])

    return results, x_vals, f_np, "secant"


def plot_newton(x_vals, f_np):
# Step 3: Plot the graph
    x_range = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 500)
    y_range = f_np(x_range)

    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_range, label="f(x)", color="blue")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)

# Plot iterations
    for i, xi in enumerate(x_vals[:-1]):
        plt.scatter(xi, f_np(xi), color="red")
        plt.plot([xi, xi], [0, f_np(xi)], color="red", linestyle="--")
        plt.plot([xi, x_vals[i + 1]], [f_np(xi), 0], color="green", linestyle="--")

    plt.title("Newton's Method Iterations")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url

def plot_secant(x_vals, f_np):
    x_range = np.linspace(min(x_vals) - 1, max(x_vals) + 1, 500)
    y_range = f_np(x_range)

    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_range, label="f(x)", color="blue")

    # Plot secant lines for each iteration
    for i in range(1, len(x_vals)):
        xi_minus1 = x_vals[i - 1]
        xi = x_vals[i]
        f_xi_minus1 = f_np(xi_minus1)
        f_xi = f_np(xi)

        # Secant line equation: y = m(x - xi_minus1) + f(xi_minus1)
        slope = (f_xi - f_xi_minus1) / (xi - xi_minus1)
        secant_y = slope * (x_range - xi_minus1) + f_xi_minus1
        plt.plot(x_range, secant_y, linestyle="--", color="green", alpha=0.6)

        # Mark points on the curve
        plt.scatter([xi_minus1, xi], [f_xi_minus1, f_xi], color="red")

    # Mark root approximation
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(x_vals[-1], color="purple", linestyle=":", linewidth=1.5, label="Approximate Root")
    plt.legend()
    plt.grid()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url


def plot_bisection(x_vals, f_np):
    # Extracting min and max from the initial intervals
    a_min = min([pair[0] for pair in x_vals])
    b_max = max([pair[1] for pair in x_vals])

    x_range = np.linspace(a_min - 1, b_max + 1, 500)
    y_range = f_np(x_range)

    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_range, label="f(x)", color="blue")

    # Plot vertical lines at each midpoint (c values)
    for i, (a, b) in enumerate(x_vals):
        c = (a + b) / 2  # Midpoint
        plt.axvline(c, color="green", linestyle="--", alpha=0.5)

    # Mark initial endpoints a and b
    plt.scatter([x_vals[0][0], x_vals[0][1]], [f_np(x_vals[0][0]), f_np(x_vals[0][1])], color="red", label="Initial Endpoints")

    # Highlight root approximation (final midpoint)
    root_approx = (x_vals[-1][0] + x_vals[-1][1]) / 2
    plt.axvline(root_approx, color="purple", linestyle=":", linewidth=1.5, label="Approximate Root")

    plt.axhline(0, color="black", linewidth=0.8)  # x-axis
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Bisection Method Iterations")
    plt.legend()
    plt.grid()

    # Convert plot to base64 for rendering in HTML
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return plot_url



@app.route('/newton', methods=['GET', 'POST'])
def newton():
    if request.method == 'POST':
        func_str = preprocess_function(request.form['function'])
        x0 = float(request.form['x0'])
        iterations = int(request.form['iterations'])
        decimal_places = int(request.form['decimal_places'])

        results, x_vals, f_np, method_name = newton_method(func_str, x0, iterations, decimal_places)
        plot_url = plot_newton(x_vals, f_np)  # Pass f_np to plot


        return render_template('result.html', results=results, plot_url=plot_url, method_name=method_name)

    return render_template('newton.html')


@app.route('/bisection', methods=['GET', 'POST'])
def bisection():
    if request.method == 'POST':
        func_str = preprocess_function(request.form['function'])
        a = float(request.form['a'])
        b = float(request.form['b'])
        iterations = int(request.form['iterations'])
        decimal_places = int(request.form['decimal_places'])

        try:
            results, x_vals, f_np, method_name = bisection_method(func_str, a, b, iterations, decimal_places)
            plot_url = plot_bisection(x_vals, f_np)  # Pass f_np to plot
            return render_template('result.html', results=results, plot_url=plot_url, method_name=method_name)
        except ValueError as e:
            return render_template('error.html', error_message=str(e))

    return render_template('bisection.html')


@app.route('/secant', methods=['GET', 'POST'])
def secant():
    if request.method == 'POST':
        func_str = preprocess_function(request.form['function'])
        x0 = float(request.form['x0'])
        x1 = float(request.form['x1'])
        iterations = int(request.form['iterations'])
        decimal_places = int(request.form['decimal_places'])

        results, x_vals, f_np, method_name = secant_method(func_str, x0, x1, iterations, decimal_places)
        plot_url = plot_secant(x_vals, f_np)  # Pass f_np to plot

        return render_template('result.html', results=results, plot_url=plot_url, method_name=method_name)

    return render_template('secant.html')


@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
