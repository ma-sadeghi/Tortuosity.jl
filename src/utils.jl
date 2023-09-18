function parse_args(s::String)
    # Parse command-line arguments in the form of "--key=value"
    regex = r"--(\w+)=([^\s]+)"
    matches = eachmatch(regex, s)
    pairs = Dict(m.captures[1] => m.captures[2] for m in matches)
    return pairs
end
