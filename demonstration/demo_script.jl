# import the required packages
import CSV
using Nro, DataFrames, CategoricalArrays, DelimitedFiles

# load the data as defined in the task guidelines
# load X as a DataFrame
X = CSV.read("./data/pima-indians-diabetes-X.csv", DataFrame; header=["x1","x2","x3","x4","x5","x6","x7","x8"])
# load y as a CategoricalVector
y = CategoricalVector(vec(readdlm("./data/pima-indians-diabetes-y.csv", Int)))

# call the oversampler
Xover, yover = naive_random_oversampler(X, y)

# print the results
show(Xover)
println()
show(IOContext(stdout, :limit => true), yover)
println()

# print the counts of the classes
println("initial count of class 0: ", count(y .== 0))
println("initial count of class 1: ", count(y .== 1))
println("count of class 0 after oversampling: ", count(yover .== 0))
println("count of class 1 after oversampling: ", count(yover .== 1))