import CSV
using DataFrames, CategoricalArrays, DelimitedFiles, Nro
X = CSV.read("/home/vmpyr/gsoc-julia/proposal/naive-random-oversampler-julia/data/pima-indians-diabetes-X.csv", DataFrame; header=["x1","x2","x3","x4","x5","x6","x7","x8"])
y = CategoricalVector(vec(readdlm("/home/vmpyr/gsoc-julia/proposal/naive-random-oversampler-julia/data/pima-indians-diabetes-y.csv", Int)))

Xnew, ynew = naive_random_oversampler(X, y)

println(Xnew)
println(ynew)