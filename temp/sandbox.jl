using Profile


function f()
    for i in 1:100000
        rand(100)
    end
end

function g()
    for i in 1:5000
        rand(100)
    end
end

function h()
    f()
    g()
end


h()
@profview h()
