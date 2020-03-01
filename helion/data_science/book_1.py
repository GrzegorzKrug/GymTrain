from sklearn.datasets import load_digits

digits = load_digits()
print(digits.DESCR)

X = digits.data
y = digits.target