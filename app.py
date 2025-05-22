import numpy as np
from sklearn.decomposition import NMF
from flask import Flask, render_template, request
import matplotlib.pyplot as plt

def factorize(V, n):
    model = NMF(n_components=n, init="random", random_state=0)
    W = model.fit_transform(V) # 725x10 , each column is a basis 
    H = model.components_
    return W, H

def generate_component(W, n):
    Wn = W[:, n]
    rW = Wn.reshape((29, 25))
    plt.imshow(rW)
    plt.savefig(f"static\\component_{n}.png")

def create_img(W, H, n):
    Wn = W[:, n]
    Hn = H[:, n]
    V = W @ H
    rV = V.reshape((29, 25))
    plt.imshow(rV)
    plt.savefig(f"static\\{n}.png")

app = Flask(__name__)

@app.route("/")
def main():
    return render_template("home.html")

@app.route("/submit", methods = ["GET", "POST"])
def spatial_components_page():
    if request.method == "POST":
        entity = request.form.get("entity")
        rank = request.form.get("rank")
    return render_template("components.html", entity=entity, rank=rank)

if __name__ == '__main__':
    app.run(debug=True)