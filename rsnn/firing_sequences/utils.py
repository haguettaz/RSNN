# coding=utf-8

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class Phis:
    def __init__(self, p):
        if p < 2:
            raise ValueError(f"The order of the polynomial must be larger than 2 but p={p} < 2")

        self.p = p
        self.phis = np.roots([1, -1] + [0] * (p - 2) + [-1])

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 12))
        sns.scatterplot(data=self.dataframe, x="re", y="im", s=50)
        sns.scatterplot(x=np.array([np.real(self.golden_number)]), y=np.array([np.imag(self.golden_number)]), s=300, marker="*")
        ax.set(xlim=(-1.7, 1.7), ylim=(-1.7, 1.7))
        ax.add_artist(plt.Circle((0, 0), 1, fill=False, linestyle="--"))

    @property
    def dataframe(self):
        return pd.DataFrame({"p": self.p, "re": np.real(self.phis), "im": np.imag(self.phis), "mod": np.abs(self.phis), "arg": np.angle(self.phis),})

    @property
    def golden_number(self):
        return np.real_if_close(self.phis[np.argmax(np.abs(self.phis))])
