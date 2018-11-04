function neighbors = k_neighbors(D, p_idx, k)
    distances = D(p_idx,:);
    [~, sorted_idxs] = sort(distances);
    neighbors = sorted_idxs(2:k+1);
end