module nro

import CSV
using DataFrames, MLUtils

# assuming 2-class classification

function naive_random_oversampler(X, y::AbstractVector)
    classes = unique(y)
    classcount = Dict(c => count(y .== c) for c in classes)
    minorityclass = argmin(classcount)
    surplus = maximum(classcount) - minimum(classcount)
    minorityidx = findall(y .== minorityclass)
    
end

end # module nro
