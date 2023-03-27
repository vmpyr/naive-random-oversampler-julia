module Nro

export naive_random_oversampler

using MLUtils

# assuming 2-class classification

function naive_random_oversampler(X, y::AbstractVector)
    classes = unique(y)
    classcount = Dict(c => count(y .== c) for c in classes)
    minorityclass = argmin(classcount)
    majorityclass = argmax(classcount)
    surplus = classcount[majorityclass] - classcount[minorityclass]
    minorityidxs = findall(y .== minorityclass)

    rndidxs = rand(minorityidxs, surplus)
    for idx in rndidxs
        push!(X, getobs(X, idx))
        push!(y, y[idx])
    end

    return X, y
end

end # module Nro
