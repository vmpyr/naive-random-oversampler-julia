module Nro

export naive_random_oversampler

# using MLUtils for getobs
using MLUtils, Tables

# assuming 2-class classification
function naive_random_oversampler(X, y::AbstractVector)
    # check if X implements getobs
    @assert Tables.istable(X) "X is not implementing the MLUtils.jl getobs interface"

    # retrieve the classes
    classes = unique(y)

    # count the number of samples in each class
    classcount = Dict(c => count(y .== c) for c in classes)

    # find the minority and majority classes
    minorityclass = argmin(classcount)
    majorityclass = argmax(classcount)

    # calculate the number of samples (surplus) to be added
    surplus = classcount[majorityclass] - classcount[minorityclass]

    # randomly select surplus number of samples from the minority class
    minorityidxs = findall(y .== minorityclass)
    rndidxs = rand(minorityidxs, surplus)

    # create a copy of the dataset
    Xover = deepcopy(X)
    yover = deepcopy(y)

    # add the samples to the copied dataset
    for idx in rndidxs
        if typeof(Xover) <: Union{Tuple, NamedTuple}
            sample = getobs(X, idx)
            for (i, col) in enumerate(Xover)
                push!(col, sample[i])
            end
        else
            push!(Xover, getobs(X, idx))
        end
        push!(yover, y[idx])
    end

    # return the new dataset
    return Xover, yover
end

end # module Nro
