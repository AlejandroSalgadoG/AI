function positions = get_n_smallest(A, n)
    [~, sorted_idxs] = sort(A);
    positions = sorted_idxs(1:n);
end